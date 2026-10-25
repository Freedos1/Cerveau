/**
 * build_report_docx.js
 * ====================
 *
 * Writes the Word (.docx) version of the report from the content blocks that
 * build_report.py records while it builds the PDF (results/report_blocks.json),
 * so both formats always carry the same text, tables and figures.
 *
 * Usage:  node build_report_docx.js results/report_blocks.json Report.docx
 *         (build_report.py calls this automatically.)
 * Needs:  npm install docx
 */

const fs = require("fs");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun,
  LevelFormat, LineRuleType, Packer, PageBreak, PageNumber, Paragraph, ShadingType, Table,
  TableCell, TableRow, TextRun, VerticalAlign, WidthType,
} = require("docx");

const [blocksPath, outPath] = process.argv.slice(2);
const blocks = JSON.parse(fs.readFileSync(blocksPath, "utf8"));

const TITLE = "Modelling Stock Price Movements with Stochastic Processes";
const BODY_FONT = "Times New Roman";
const HEAD_FONT = "Arial";
const INK = "0B0B0B";
const MUTED = "52514E";
const HEAD_BG = "F1F0EC";
const CM = 567;                       // twentieths of a point (DXA) per centimetre
const CONTENT_WIDTH = 16.6 * CM;      // A4 width minus 2.2 cm margins
const PX_PER_CM = 96 / 2.54;          // ImageRun sizes are in pixels at 96 dpi

// ---- inline markup -> TextRuns -------------------------------------------
// build_report.py writes ReportLab mini-HTML: <b>, <i>, <sub>, <super>,
// <font face='Mono'> and HTML entities.
const ENTITIES = {
  amp: "&", lt: "<", gt: ">", ndash: "–", mdash: "—", minus: "−",
  mu: "μ", sigma: "σ", kappa: "κ", theta: "θ", lambda: "λ",
  Delta: "Δ", chi: "χ", epsilon: "ε", radic: "√", asymp: "≈",
  middot: "·", hellip: "…", lsquo: "‘", rsquo: "’", iacute: "í",
};
const decode = (s) => s.replace(/&([a-zA-Z]+|#\d+);/g, (m, name) =>
  name.startsWith("#") ? String.fromCodePoint(+name.slice(1)) : (ENTITIES[name] ?? m));

function runs(markup, base = {}) {
  const out = [];
  const state = { bold: false, italics: false, sub: false, sup: false, mono: false };
  const tokens = markup.split(/(<\/?(?:b|i|sub|super)>|<font face='Mono'>|<\/font>)/);
  for (const tok of tokens) {
    switch (tok) {
      case "<b>": state.bold = true; continue;
      case "</b>": state.bold = false; continue;
      case "<i>": state.italics = true; continue;
      case "</i>": state.italics = false; continue;
      case "<sub>": state.sub = true; continue;
      case "</sub>": state.sub = false; continue;
      case "<super>": state.sup = true; continue;
      case "</super>": state.sup = false; continue;
      case "<font face='Mono'>": state.mono = true; continue;
      case "</font>": state.mono = false; continue;
    }
    if (!tok) continue;
    out.push(new TextRun({
      ...base,
      text: decode(tok),
      bold: base.bold || state.bold,
      italics: base.italics || state.italics,
      subScript: state.sub,
      superScript: state.sup,
      ...(state.mono ? { font: "Consolas", size: (base.size ?? 21) - 2 } : {}),
    }));
  }
  return out;
}

// ---- block builders -----------------------------------------------------------
const P = (markup, opts = {}, runOpts = {}) =>
  new Paragraph({ children: runs(markup, runOpts), ...opts });

const caption = (markup) => P(markup,
  { alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 200 } },
  { italics: true, size: 18, color: MUTED });

function figure(b) {
  const widthPx = b.width_cm * PX_PER_CM;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true,
      spacing: { before: 120, line: 240, lineRule: LineRuleType.AUTO },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(b.path),
        transformation: { width: widthPx, height: widthPx * b.aspect },
        altText: { title: "Figure", description: b.caption.replace(/<[^>]+>/g, ""), name: "figure" },
      })],
    }),
    caption(b.caption),
  ];
}

function table(b) {
  const nCols = b.rows[0].length;
  let widths = b.widths_cm ? b.widths_cm.map((w) => Math.round(w * CM))
                           : Array(nCols).fill(Math.floor(CONTENT_WIDTH / nCols));
  const total = widths.reduce((a, c) => a + c, 0);
  const rule = (size) => ({ style: BorderStyle.SINGLE, size, color: INK });
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const rows = b.rows.map((row, i) => {
    const header = i < b.header_rows;
    return new TableRow({
      tableHeader: header,
      cantSplit: true,
      children: row.map((txt, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        verticalAlign: VerticalAlign.CENTER,
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        shading: header ? { type: ShadingType.CLEAR, color: "auto", fill: HEAD_BG }
                        : (i % 2 === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "FAFAF8" } : undefined),
        borders: {
          top: i === 0 ? rule(8) : none,
          bottom: (i === b.header_rows - 1) ? rule(6) : (i === b.rows.length - 1 ? rule(8) : none),
          left: none, right: none,
        },
        children: [P(txt, {}, { font: HEAD_FONT, size: 16, bold: header })],
      })),
    });
  });
  return [
    P(b.caption, { keepNext: true, spacing: { before: 160, after: 80 } }, { size: 18 }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
                alignment: AlignmentType.CENTER, rows }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

const children = [];
for (const b of blocks) {
  switch (b.type) {
    case "title":
      children.push(P(b.text, { alignment: AlignmentType.CENTER, spacing: { before: 600, after: 160 } },
                      { font: HEAD_FONT, bold: true, size: 38 }));
      break;
    case "subtitle":
      children.push(P(b.text, { alignment: AlignmentType.CENTER, spacing: { after: 80 } },
                      { font: HEAD_FONT, size: 22, color: MUTED }));
      break;
    case "h1":
      children.push(P(b.text, { heading: HeadingLevel.HEADING_1, keepNext: true }));
      break;
    case "h2":
      children.push(P(b.text.replace(/<\/?b>/g, ""), { heading: HeadingLevel.HEADING_2, keepNext: true }));
      break;
    case "eq":
      children.push(P(b.text, { alignment: AlignmentType.CENTER, spacing: { before: 40, after: 120 } },
                      { italics: true }));
      break;
    case "bullet":
      children.push(P(b.text, { numbering: { reference: "bullets", level: 0 },
                                alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 } }));
      break;
    case "ref":
      children.push(P(b.text, { indent: { left: 360, hanging: 360 }, spacing: { after: 80 } },
                      { size: 20 }));
      break;
    case "figure": children.push(...figure(b)); break;
    case "table": children.push(...table(b)); break;
    case "pagebreak": children.push(new Paragraph({ children: [new PageBreak()] })); break;
    default:   // body text
      children.push(P(b.text, { alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 } }));
  }
}

const doc = new Document({
  title: TITLE,
  creator: "Quantitative Research",
  styles: {
    default: { document: { run: { font: BODY_FONT, size: 21, color: INK },
                           paragraph: { spacing: { line: 276, lineRule: LineRuleType.AUTO } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 27, bold: true, color: INK },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 22, bold: true, color: INK },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{ reference: "bullets", levels: [{
      level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 400, hanging: 260 } } },
    }] }],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 },          // A4
              margin: { top: 1134, bottom: 1134, left: 1247, right: 1247 } },
      titlePage: true,
    },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: TITLE, font: HEAD_FONT, size: 15, color: MUTED })],
    })] }) },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: HEAD_FONT, size: 16, color: MUTED })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log(`Word report written to ${outPath}`);
});
