import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

HEADINGS = {
    "discussion": "Обсуждение",
    "proposal": "Предложения",
    "decision": "Принятые решения",
    "task": "Поручения",
    "open_question": "Открытые рабочие вопросы",
    "risk": "Риски и ограничения",
}


def clean(value):
    return "".join(
        c for c in str(value)
        if c in "\n\t" or (
            ord(c) >= 32
            and not 0xD800 <= ord(c) <= 0xDFFF
            and ord(c) not in (0xFFFE, 0xFFFF)
        )
    )


def timestamp(seconds):
    milliseconds = round(float(seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def main():
    parser = argparse.ArgumentParser(
        description="DOCX и PDF из facts.json без обращения к моделям."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--speaker-names", type=Path)
    parser.add_argument("--font", type=Path, help="TTF-шрифт с кириллицей для PDF")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from docx import Document
    from docx.shared import Mm, Pt
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, PageBreak
    )

    source = args.source.resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    display_names = {}
    if args.speaker_names:
        names_data = json.loads(
            args.speaker_names.read_text(encoding="utf-8")
        )
        facts_source = data.get("source_transcript")
        names_source = names_data.get("source_transcript")
        if not facts_source or not names_source:
            parser.error("Нет ссылки на исходный транскрипт")
        if Path(facts_source).resolve() != Path(names_source).resolve():
            parser.error("Факты и таблица имён относятся к разным транскриптам")
        if names_data.get("status") != "needs_review":
            parser.error("Неожиданный статус таблицы имён")

        for speaker, entry in names_data.get("speakers", {}).items():
            name = entry.get("name")
            if entry.get("status") == "inferred" and isinstance(name, str):
                if name.strip():
                    display_names[speaker.casefold()] = (
                        name.strip() + " (предположительно)"
                    )

    def render_names(value):
        return re.sub(
            r"\bspeaker_\d+\b",
            lambda match: display_names.get(
                match.group(0).casefold(), match.group(0)
            ),
            value,
            flags=re.IGNORECASE,
        )

    items = data.get("items")
    if not isinstance(items, list):
        parser.error("Ожидается объект с массивом items")

    for item in items:
        if item.get("kind") not in HEADINGS:
            parser.error(f"Неизвестная категория: {item.get('kind')}")
        if not isinstance(item.get("text"), str):
            parser.error("У пункта отсутствует текст")

    candidates = [
        args.font,
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/mnt/c/Windows/Fonts/arial.ttf"),
    ]
    font = next(
        (p for p in candidates if p is not None and p.is_file()), None
    )
    if font is None:
        parser.error("Не найден шрифт. Укажите --font /путь/к/шрифту.ttf")

    out = (args.out or source.parent / "reports").resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / "protocol.docx", out / "protocol.pdf"]
    if not args.overwrite and any(p.exists() for p in targets):
        parser.error("Документы уже существуют. Укажите новую папку или --overwrite")
    if source in targets:
        parser.error("Выходной документ совпадает с входом")

    document = Document()
    section = document.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.top_margin = section.bottom_margin = Mm(20)
    section.left_margin = section.right_margin = Mm(20)
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(11)

    pdfmetrics.registerFont(TTFont("ReportFont", str(font)))
    styles = {
        "title": ParagraphStyle(
            "ReportTitle", fontName="ReportFont", fontSize=18,
            leading=23, spaceAfter=14,
        ),
        "heading": ParagraphStyle(
            "ReportHeading", fontName="ReportFont", fontSize=14,
            leading=19, spaceBefore=12, spaceAfter=8, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody", fontName="ReportFont", fontSize=10,
            leading=15, spaceAfter=7, splitLongWords=True,
        ),
        "source": ParagraphStyle(
            "ReportSource", fontName="ReportFont", fontSize=9,
            leading=13, spaceAfter=7, splitLongWords=True,
        ),
    }
    story = []

    def add(text, kind="body"):
        text = clean(text)
        if kind == "body":
            text = render_names(text)
        if kind == "title":
            document.add_heading(text, 0)
        elif kind == "heading":
            document.add_heading(text, 1)
        else:
            document.add_paragraph(text)
        html = escape(text).replace("\n", "<br/>")
        story.append(Paragraph(html, styles[kind]))

    add("Протокол — черновик", "title")
    add(f"Встреча: {data.get('meeting_id') or 'не указана'}")
    add(f"Статус: {data.get('status', 'не указан')}")
    add("Требует проверки. Не является утверждённым протоколом.")
    if display_names:
        add(
            "Имена определены предположительно по разговору. "
            "Личности участников не подтверждены. "
            "Исходные идентификаторы сохранены в приложении."
        )

    failed = data.get("failed_blocks", [])
    if failed or data.get("status") == "partial_needs_review":
        add(
            "ВНИМАНИЕ: документ неполный. "
            f"Необработанных блоков: {len(failed)}."
        )

    for warning in data.get("warnings", []):
        add("Предупреждение: " + str(warning.get("warning", warning)))

    sources = {}
    item_references = []
    for item in items:
        references = []
        for evidence in item.get("evidence", []):
            key = (
                evidence["utterance_id"],
                evidence.get("quote", ""),
                evidence.get("source_scope", ""),
            )
            if key not in sources:
                sources[key] = {
                    "number": len(sources) + 1,
                    "evidence": evidence,
                }
            references.append(sources[key]["number"])
        item_references.append(list(dict.fromkeys(references)))

    for kind, heading in HEADINGS.items():
        add(heading, "heading")
        selected = [
            (index, item)
            for index, item in enumerate(items) if item["kind"] == kind
        ]
        if not selected:
            add("В обработанных блоках автоматически не извлечены.")
            continue
        for index, item in selected:
            references = item_references[index]
            suffix = (
                " [Источники: " + ", ".join(map(str, references)) + "]"
                if references else " [Источник не указан]"
            )
            add(item["text"] + suffix)
            if kind == "task":
                add(
                    f"Исполнитель: {item.get('owner') or 'не установлен'}\n"
                    f"Срок: {item.get('deadline') or 'не установлен'}"
                )

    limitations = data.get("limitations", [])
    if limitations:
        add("Ограничения", "heading")
        for limitation in limitations:
            add("— " + str(limitation))

    if sources:
        document.add_page_break()
        story.append(PageBreak())
        add("Приложение: источники", "heading")
        add(
            "Источники воспроизводят текст ASR. "
            "Он может содержать ошибки распознавания."
        )
        for entry in sources.values():
            evidence = entry["evidence"]
            scope = (
                "полная реплика"
                if evidence.get("source_scope") == "full_utterance"
                else "фрагмент реплики"
            )
            start, end = evidence.get("start"), evidence.get("end")
            timing = (
                f"{timestamp(start)}–{timestamp(end)}"
                if start is not None and end is not None
                else "таймкоды не указаны"
            )
            add(
                f"[{entry['number']}] {scope} {evidence['utterance_id']}; "
                f"{evidence.get('speaker', 'неизвестный спикер')}; {timing}",
                "source",
            )
            add(evidence.get("quote", ""), "source")
            flags = evidence.get("source_review_flags", [])
            if flags:
                add("Отметки проверки: " + ", ".join(flags), "source")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("ReportFont", 8)
        canvas.drawRightString(A4[0] - 40, 25, str(doc.page))
        canvas.restoreState()

    with tempfile.TemporaryDirectory(dir=out) as temp:
        temporary_docx = Path(temp) / "protocol.docx"
        temporary_pdf = Path(temp) / "protocol.pdf"
        document.save(temporary_docx)
        SimpleDocTemplate(
            str(temporary_pdf), pagesize=A4,
            rightMargin=40, leftMargin=40,
            topMargin=40, bottomMargin=40,
        ).build(story, onFirstPage=footer, onLaterPages=footer)

        for temporary in [temporary_docx, temporary_pdf]:
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("Не создан документ: " + temporary.name)

        for temporary, target in zip(
            [temporary_docx, temporary_pdf], targets
        ):
            os.replace(temporary, target)

    for target in targets:
        print(f"Сохранено: {target} ({target.stat().st_size} байт)")


if __name__ == "__main__":
    main()
