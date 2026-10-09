// Builds the Assignment 4 paper (APA 7 student format) as a .docx file.
// Usage: node build_paper.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Header, AlignmentType, PageNumber,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, TabStopType,
  HeadingLevel, LevelFormat, PageBreak,
} = require("docx");

const FONT = "Times New Roman";
const MONO = "Courier New";
const DOUBLE = 480;
const FIG = path.join(__dirname, "figures");
const CODE = path.join(__dirname, "code");

// ---------- inline text markup: **bold**, *italic*, _{sub}, ^{sup}, `mono` ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\}|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("***")) out.push(new TextRun({ text: t.slice(3, -3), bold: true, italics: true, ...base }));
    else if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, italics: true, ...base }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, italics: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: 22, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text), spacing: { line: DOUBLE }, indent: { firstLine: 720 }, ...opts,
});
const PN = (text, opts = {}) => P(text, { indent: { firstLine: 0 }, ...opts });
const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text, bold: true })], spacing: { line: DOUBLE },
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, bold: true })],
  spacing: { line: DOUBLE },
});
const H3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3, children: [new TextRun({ text, bold: true, italics: true })],
  spacing: { line: DOUBLE },
});
const bullet = (text, ref = "bullets") => new Paragraph({
  children: runs(text), numbering: { reference: ref, level: 0 }, spacing: { line: DOUBLE },
});
const numbered = (text, ref) => bullet(text, ref);
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// Display equation (pre-rendered PNG, see code/render_equations.py), centered, numbered.
function EQ(num) {
  const data = fs.readFileSync(path.join(FIG, "equations", `eq_${num}.png`));
  const scale = 96 / 300;            // rendered at 300 dpi; docx sizes are in 96-dpi pixels
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: 4680 }, { type: TabStopType.RIGHT, position: 9360 }],
    spacing: { before: 120, after: 120 },
    children: [
      new TextRun("\t"),
      new ImageRun({ type: "png", data, transformation: { width: Math.round(data.readUInt32BE(16) * scale), height: Math.round(data.readUInt32BE(20) * scale) },
        altText: { title: `Equation ${num}`, description: `Equation ${num}`, name: `eq${num}` } }),
      new TextRun(`\t(${num})`),
    ],
  });
}

// ---------- code block ----------
function codeBlock(lines) {
  return lines.map((l, idx) => new Paragraph({
    children: [new TextRun({ text: l.length ? l : " ", font: MONO, size: 17 })],
    spacing: { line: 240, before: idx === 0 ? 120 : 0, after: idx === lines.length - 1 ? 240 : 0 },
    shading: { type: ShadingType.CLEAR, fill: "F3F3F3", color: "auto" },
    indent: { left: 240, right: 240 },
  }));
}
function fileLines(file, from, to) {
  return fs.readFileSync(path.join(CODE, file), "utf8").split("\n").slice(from - 1, to);
}

// ---------- APA table (three horizontal rules, no vertical lines) ----------
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
function apaTable(number, title, header, rows, widths, note) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, { top = NONE, bottom = NONE, bold = false } = {}) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top, bottom, left: NONE, right: NONE },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(text, bold ? { bold: true } : {}), spacing: { line: 240 } })],
  });
  const tRows = [new TableRow({ tableHeader: true, children: header.map((h, j) => cell(h, widths[j], { top: RULE, bottom: RULE })) })];
  rows.forEach((r, ri) => tRows.push(new TableRow({
    children: r.map((c, j) => cell(c, widths[j], { bottom: ri === rows.length - 1 ? RULE : NONE })),
  })));
  const out = [
    new Paragraph({ children: [new TextRun({ text: `Table ${number}`, bold: true })], spacing: { line: DOUBLE, before: 240 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: title, italics: true })], spacing: { line: DOUBLE }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: tRows }),
  ];
  if (note) out.push(new Paragraph({ children: [new TextRun({ text: "Note. ", italics: true }), ...runs(note)], spacing: { line: DOUBLE, after: 240 } }));
  else out.push(new Paragraph({ children: [], spacing: { after: 120 } }));
  return out;
}

// ---------- APA figure ----------
function figure(number, title, file, w, h, note) {
  return [
    new Paragraph({ children: [new TextRun({ text: `Figure ${number}`, bold: true })], spacing: { line: DOUBLE, before: 240 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: title, italics: true })], spacing: { line: DOUBLE }, keepNext: true }),
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)), transformation: { width: w, height: h },
        altText: { title: `Figure ${number}`, description: title, name: `figure${number}` } })],
    }),
    new Paragraph({ children: [new TextRun({ text: "Note. ", italics: true }), ...runs(note)], spacing: { line: DOUBLE, after: 240 } }),
  ];
}

// ---------- references (hanging indent) ----------
const REF = (text) => new Paragraph({ children: runs(text), spacing: { line: DOUBLE }, indent: { left: 720, hanging: 720 } });

// =====================================================================
// CONTENT
// =====================================================================
const titlePage = [
  ...Array.from({ length: 3 }, () => new Paragraph({ children: [], spacing: { line: DOUBLE } })),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: DOUBLE },
    children: [new TextRun({ text: "Memory Networks for Question Answering: Mathematical Foundations,", bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: DOUBLE },
    children: [new TextRun({ text: "Relation Extraction, and Graph-Based Reasoning", bold: true })] }),
  new Paragraph({ children: [], spacing: { line: DOUBLE } }),
  ...["[Student Name]", "[Department], [University]", "[Course Number]: [Course Name]", "[Instructor Name]", "October 9, 2026"]
    .map((t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: DOUBLE }, children: [new TextRun(t)] })),
  pageBreak(),
];

const abstract = [
  H1("Abstract"),
  PN("Memory networks (Weston et al., 2015) pair a learning model with a long-term memory that the model can write to and read from. This paper explains how they work. Part A covers the memory, the four components *I*, *G*, *O*, and *R*, the flow of data through the model, the text implementation called MemNN, and its training with a margin ranking loss. Part B shows how a pre-trained relation extractor, REBEL (Huguet Cabot & Navigli, 2021), can turn a question and its answer choices into subject–relation–object triplets. Part C uses the question “Why did the engine stop?” and the choice “The fuel tank was empty.” to show how those triplets become a small graph, how a relational graph neural network walks the path from the question to the answer, and how the question text is fused with the graph. The paper ends by linking the two ideas: a graph neural network over retrieved facts does the same job as the *O* module of a memory network, but it follows labeled relations instead of guessing which stored sentences belong together."),
  new Paragraph({ spacing: { line: DOUBLE }, indent: { firstLine: 720 },
    children: [new TextRun({ text: "Keywords: ", italics: true }), new TextRun("memory networks, question answering, relation extraction, knowledge graphs, graph neural networks")] }),
  pageBreak(),
];

const intro = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: DOUBLE },
    children: [new TextRun({ text: "Memory Networks for Question Answering: Mathematical Foundations, Relation Extraction, and Graph-Based Reasoning", bold: true })] }),
  P("To answer a question about a story, a system must remember the facts in the story and then combine the right ones. Recurrent neural networks (RNNs) and long short-term memory networks (LSTMs; Hochreiter & Schmidhuber, 1997) store everything they have read in one hidden state vector. That vector is small, and old facts fade or get overwritten. Weston et al. (2015) proposed *memory networks* to fix this. A memory network keeps an explicit memory, a list of stored items, and learns how to write to it, how to look things up in it, and how to turn what it found into an answer."),
  P("This paper has three parts. Part A explains the model in the original paper and the math behind it. Part B shows how a pre-trained relation extractor can turn questions and answer choices into structured triplets. Part C shows how those triplets can form a small graph and how a graph neural network (GNN) can reason over it, using the question “Why did the engine stop?” as a running example. Runnable Python versions of the Part B and Part C examples come with this paper (files `relation_extraction_demo.py` and `graph_reasoning_demo.py`)."),
];

// ---------------- PART A ----------------
const partA = [
  H1("Part A: Memory Networks"),
  P("Weston et al. (2015) describe memory networks first as a general framework and then as one concrete neural version for text, called a MemNN. The framework says *what* the parts must do. The MemNN says *how* one simple version does it."),

  H2("Memory"),
  P("The memory **m** is an array of objects indexed by *i*, written **m**_{i}. The objects can be vectors or raw strings (Weston et al., 2015, p. 1). In the text version, each slot holds one sentence in its original form, such as “Joe went to the kitchen.” The memory has three key properties."),
  bullet("**It is compartmentalized.** Each fact sits in its own slot. This differs from an RNN, which compresses all facts into one dense vector, and from classic associative memories, which spread knowledge across all the weights of a network (Weston et al., 2015, p. 6)."),
  bullet("**It can be read and written.** The model adds new facts at both training and test time. The model *parameters* stay fixed at test time, but the *memory contents* keep changing. This is how the model can answer questions about a story it has never seen."),
  bullet("**It can be very large.** In the paper’s large-scale experiment, the memory holds 14 million facts, stored as (subject, relation, object) triples from the REVERB extractions used by Fader et al. (2013)."),
  P("The paper also discusses how to manage a large memory. A slot-choosing function *H* decides where each new item goes. *H* could group memories by topic or entity, so later lookups only search the right group. If the memory fills up, *H* could also “forget” by overwriting the least useful slot, though the authors did not test this."),

  H2("Components"),
  P("A memory network has four components. Each can be learned, and each can, in principle, use any machine learning method (Weston et al., 2015, p. 2)."),
  bullet("***I*** **(input feature map)** turns the raw input *x* into an internal representation *I*(*x*). For text, this can include parsing, coreference resolution, or mapping words to a bag-of-words vector."),
  bullet("***G*** **(generalization)** updates the memory using the new input. The simplest *G* stores *I*(*x*) in a free slot and leaves all other slots alone:"),
  EQ(1),
  PN("where *H*(*x*) picks the slot. A richer *G* could revise old memories based on new evidence. The authors call it “generalization” because this is where the network could compress and generalize what it stores."),
  bullet("***O*** **(output feature map)** reads the memory. Given the input and the current memory, it finds the relevant memories and produces output features *o*. This is where inference happens."),
  bullet("***R*** **(response)** turns *o* into the final answer, such as a single word, a sentence produced by an RNN, or an action."),

  H2("Flow of the Model"),
  P("Given an input *x* (a word, a sentence, an image, and so on), the model runs four steps (Weston et al., 2015, p. 2):"),
  numbered("Convert *x* to an internal feature representation *I*(*x*).", "flow"),
  numbered("Update the memories given the new input: **m**_{i} = *G*(**m**_{i}, *I*(*x*), **m**) for all *i*.", "flow"),
  numbered("Compute output features from the input and the memory: *o* = *O*(*I*(*x*), **m**).", "flow"),
  numbered("Decode the output features into a response: *r* = *R*(*o*).", "flow"),
  P("Figure 1 shows this flow. A statement such as “Joe went to the kitchen” mostly uses the write path (*I* then *G*). A question such as “Where is the milk now?” mostly uses the read path (*I*, then *O*, then *R*). The same steps run during training and at test time."),
  ...figure(1, "Data Flow Through a Memory Network", "fig1_memnn_flow.png", 600, 300,
    "*I*, *G*, *O*, and *R* are the four components defined by Weston et al. (2015). *G* writes the encoded input into memory; *O* scores memories against the input and selects supporting facts; *R* produces the answer. Figure created by the author."),

  H2("Implementation of a Memory Network for Text"),
  P("A memory network whose components are neural networks is called a memory neural network, or MemNN. Section 3 of the paper builds a simple MemNN for text (Weston et al., 2015, pp. 2–6)."),
  H3("Input and Generalization"),
  P("*I* takes a sentence, which is either a fact or a question. *G* stores it in the next empty slot, so **m**_{N} = *x* and then *N* = *N* + 1. Old memories are never changed."),
  H3("Output: Finding Supporting Memories"),
  P("The core of the model is the *O* module. It finds *k* supporting memories for the input *x*; the paper uses *k* up to 2. The first supporting memory is the one that best matches the question:"),
  EQ(2),
  PN("where *s*_{O} is a learned scoring function. For *k* = 2, the model looks for a second memory that matches both the question and the first memory found:"),
  EQ(3),
  PN("The square brackets denote a list. Because inputs are bags of words, scoring the list is the same as adding *s*_{O}(*x*, **m**_{i}) and *s*_{O}(**m**_{o1}, **m**_{i}). The output is *o* = [*x*, **m**_{o1}, **m**_{o2}]. This two-step search is called a *hop*. Each hop lets the model chain one more fact."),
  H3("Response"),
  P("The simplest *R* returns the retrieved sentence **m**_{ok}. The paper’s main version instead returns a single word by ranking every word *w* in the dictionary *W*:"),
  EQ(4),
  PN("An RNN or LSTM can replace this step when full sentences are needed."),
  H3("Scoring Function"),
  P("Both *s*_{O} and *s*_{R} have the same form, an embedding model:"),
  EQ(5),
  PN("Here Φ_{x} and Φ_{y} map text to a *D*-dimensional sparse feature vector (a bag of words), and *U* is an *n* × *D* matrix that projects those features into an *n*-dimensional embedding space. The score is the dot product of the two embeddings, *U*Φ_{x}(*x*) and *U*Φ_{y}(*y*). The paper sets *D* = 3|*W*|. Each word gets three separate representations: one when it appears in the candidate *y*, and two for the input side, depending on whether it comes from the question or from an already retrieved memory. *s*_{O} and *s*_{R} use different matrices, *U*_{O} and *U*_{R}."),
  H3("A Worked Example"),
  P("Take the story in Figure 1 of the original paper: “Joe went to the kitchen. Fred went to the kitchen. Joe picked up the milk. Joe travelled to the office. Joe left the milk. Joe went to the bathroom.” The question is *x* = “Where is the milk now?”"),
  bullet("Hop 1 (Equation 2): the best match for *x* is **m**_{o1} = “Joe left the milk,” because it shares the word *milk*."),
  bullet("Hop 2 (Equation 3): given [*x*, **m**_{o1}], the best match is **m**_{o2} = “Joe travelled to the office,” the last place Joe went before leaving the milk."),
  bullet("Response (Equation 4): ranking all words given [*x*, **m**_{o1}, **m**_{o2}] gives *r* = “office.”"),
  P("Note that “office” does not appear in the sentence that mentions the milk. The model reaches it only by chaining two facts. This is what “reasoning over memory” means here."),
  H3("Extensions"),
  P("The paper adds four extensions to the basic model (Weston et al., 2015, pp. 4–6)."),
  bullet("**Word sequences as input.** When text arrives as a stream of words, a learned segmenter decides where a statement ends: *seg*(*c*) = *W*_{seg}^{⊤}*U*_{S}Φ_{seg}(*c*). If the score passes the margin γ, the words so far are written to memory as one slot."),
  bullet("**Hashing for speed.** Scoring 14 million memories for each question is slow. The paper hashes the input into buckets and only scores memories in the same buckets. Hashing by word is fast but misses memories with no shared word. Hashing by clusters of word embeddings (*K*-means on the columns of *U*_{O}) also catches synonyms."),
  bullet("**Write time.** In stories, the order of facts matters. Instead of scoring one memory at a time, the model compares two candidate memories *y* and *y*′:"),
  EQ(6),
  PN("Φ_{t} adds three binary features that say whether *x* is older than *y*, whether *x* is older than *y*′, and whether *y* is older than *y*′. A positive score means *y* is preferred. The argmax becomes a loop that keeps the current winner and compares it with each next memory."),
  bullet("**Unseen words and exact matches.** A new word (the paper uses “Boromir”) is represented by the words that appear around it, which raises *D* to 5|*W*|. Because low-dimensional embeddings blur exact word matches, the paper also adds matching features that mark words shared by *x* and *y* (giving *D* = 8|*W*|), or mixes in a raw bag-of-words score:"),
  EQ(7),

  H2("Training Procedure for a Memory Network"),
  P("The MemNN is trained with full supervision. For each training question, the data give the correct answer *r* and also mark which sentences are the supporting facts, **m**_{o1} and **m**_{o2}. The supporting facts are *not* given at test time. Knowing the right choices for both argmax steps during training allows a simple loss. The model minimizes, over the parameters *U*_{O} and *U*_{R}, a sum of three margin ranking (hinge) terms (Weston et al., 2015, p. 4):"),
  EQ("8a"),
  EQ("8b"),
  EQ("8c"),
  P("Here f̄, f̄′, and r̄ are wrong choices and γ is the margin. Each term says the same thing: the correct choice must score at least γ higher than any wrong choice. If it already does, the term is zero and nothing changes. If it does not, the gradient pushes the correct score up and the wrong score down. Term 8a trains the first hop, term 8b trains the second hop, and term 8c trains the answer ranking."),
  P("The model is trained with stochastic gradient descent (SGD). Summing over every wrong choice would be expensive, so at each step the model samples one f̄, f̄′, and r̄, as in the WSABIE method of Weston et al. (2011). In the simulated-world experiments, the authors used an embedding size of 100, a learning rate of 0.01, a margin γ of 0.1, and 10 training epochs. When an RNN serves as *R*, term 8c is replaced by the usual log-likelihood loss for language modeling on the sequence [*x*, *o*_{1}, *o*_{2}, *r*]. The segmenter and the write-time model have their own hinge losses of the same kind (Weston et al., 2015, Appendices B and C)."),
  P("Table 1 summarizes the main results. On the large-scale QA task, the MemNN matched earlier embedding methods, and adding bag-of-words features raised F1 from 0.72 to 0.82. Cluster hashing gave about an 80-fold speedup with almost no loss. On the simulated stories, the version with two hops and time features answered nearly every question, while RNNs and LSTMs failed on the harder tasks."),
  ...apaTable(1, "Selected Results Reported by Weston et al. (2015)",
    ["Task", "Model", "Result"],
    [["Large-scale QA (Fader et al., 2013)", "Fader et al. (2013)", "F1 = 0.54"],
     ["", "Bordes et al. (2014)", "F1 = 0.73"],
     ["", "MemNN, embedding only", "F1 = 0.72"],
     ["", "MemNN, with bag-of-words features", "F1 = 0.82"],
     ["", "MemNN, cluster hash (80× faster)", "F1 = 0.80"],
     ["Simulated world, difficulty 5, actor + object", "RNN", "17.8% accuracy"],
     ["", "LSTM", "29.0% accuracy"],
     ["", "MemNN, *k* = 1", "18.5% accuracy"],
     ["", "MemNN, *k* = 2, with time features", "99.9% accuracy"]],
    [3500, 3700, 2160],
    "Values from Tables 1, 2, and 3 of Weston et al. (2015). “Difficulty 5” means the needed fact may be up to five steps back in the story."),
  P("Later work kept the idea but changed the training. End-to-end memory networks replace the hard argmax with a soft attention over all memories, so the model can learn from answers alone, without labels for supporting facts (Sukhbaatar et al., 2015). Dynamic memory networks add an episodic memory module with gated recurrent updates (Kumar et al., 2016). The bAbI tasks, built from the same kind of simulated world, became a standard test for this family of models (Weston et al., 2016)."),
];

// ---------------- PART B ----------------
const partB = [
  H1("Part B: Relation Extraction"),
  P("Relation extraction turns free text into triplets of the form (subject, relation, object). Early systems used hand-written patterns or open extraction, which keeps the verb phrase as the relation (Banko et al., 2007). Later systems used distant supervision, which labels training sentences by matching them to facts already in a knowledge base (Mintz et al., 2009). Modern pre-trained models learn the whole task end to end. REBEL (Huguet Cabot & Navigli, 2021) is one such model. It is a sequence-to-sequence transformer (BART) fine-tuned to read a sentence and write out its triplets in a fixed format that uses over 200 relation types from Wikidata:"),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: DOUBLE },
    children: [new TextRun({ text: "<triplet> subject <subj> object <obj> relation", font: MONO, size: 21 })] }),
  P("This link to memory networks is direct. The large-scale experiment in Weston et al. (2015) stored 14 million triplets that an extraction system had already pulled from the web. Relation extraction is how those memories are made."),
  H2("Step 1: Prepare the Inputs"),
  P("A question and its answer choices are short and often do not state a full fact. “Why did the engine stop?” asks for a cause; it does not state one. For this reason, the example runs the extractor on three kinds of text: (a) the question alone, (b) each answer choice alone, and (c) one declarative *hypothesis* per choice that joins the two, such as “The engine stopped because the fuel tank was empty.” The question word also gives a cue: *why* signals that the missing link is a cause."),
  H2("Step 2: Run the Pre-trained Extractor"),
  P("The code below loads REBEL through the Hugging Face `transformers` library and parses its output into Python tuples. REBEL’s special tokens must be kept when decoding, because they mark where each part of a triplet starts."),
  ...codeBlock([
    ...fileLines("relation_extraction_demo.py", 9, 32),
    "",
    ...fileLines("relation_extraction_demo.py", 42, 55),
  ]),
  H2("Step 3: Read and Normalize the Triplets"),
  P("Table 2 shows the kind of output this pipeline gives for the example. Raw relation names from REBEL (Wikidata labels such as *has cause* and *has effect*) and verb phrases from an open extractor are then mapped to a small, fixed set of relations that the graph in Part C uses. Mapping to the relation names of ConceptNet (Speer et al., 2017) makes it easy to link the triplets to background knowledge in the next step."),
  ...apaTable(2, "Triplets Extracted From the Question and Answer Choices",
    ["Source text", "Raw triplet (subject | relation | object)", "Normalized triplet"],
    [["Q: Why did the engine stop?", "engine | has effect | stop", "(engine, HasSubevent, stop); target relation = Causes, object = ?"],
     ["A: The fuel tank was empty.", "fuel tank | has quality | empty", "(fuel tank, HasState, empty)"],
     ["B: The driver turned on the radio.", "driver | uses | radio", "(driver, Uses, radio)"],
     ["C: The tires were new.", "tires | has quality | new", "(tires, HasProperty, new)"],
     ["Hypothesis A: The engine stopped because the fuel tank was empty.", "fuel tank was empty | has effect | engine stopped", "claim: (empty fuel tank, Causes, engine stop)"],
     ["Hypothesis B: The engine stopped because the driver turned on the radio.", "radio | has effect | engine stopped", "claim: (radio, Causes, engine stop)"]],
    [3000, 3260, 3100],
    "Exact strings depend on the model version and decoding settings; the rows show the expected shape of the output. Hypothesis rows are claims to be checked, not evidence."),
  P("One point in Table 2 needs care. The extractor copies whatever a sentence claims. It pulls a causal triplet from Hypothesis B just as readily as from Hypothesis A, even though turning on a radio does not stop an engine. So the hypothesis triplets cannot decide the answer by themselves. Their job is to name the entities and the relation the answer must support. The evidence must come from somewhere else, namely a knowledge graph of background facts, and the next part uses a GNN to test whether that evidence connects the question to each choice."),
  H2("Step 4: Ground the Entities in a Knowledge Graph"),
  P("Each extracted subject and object is matched to a node in a commonsense knowledge graph, such as ConceptNet (Speer et al., 2017) or ATOMIC, which stores everyday cause-and-effect knowledge (Sap et al., 2019). This is the grounding step used by KagNet (Lin et al., 2019) and QA-GNN (Yasunaga et al., 2021). For the example, grounding finds *engine* and *stop* for the question, *fuel tank* and *empty* for choice A, and pulls in neighboring background facts such as (engine, Uses, fuel), (fuel tank, Contains, fuel), (empty, Implies, lack of fuel), and (lack of fuel, Causes, stop). These are the edges of the sub-graph in Part C."),
];

// ---------------- PART C ----------------
const partC = [
  H1("Part C: Graph-Based Reasoning"),
  P("The question is “Why did the engine stop?” and the answer choice is “The fuel tank was empty.” The two sentences share no content words. A system that only matches words, like the basic MemNN scoring function in Equation 5, has nothing to match. The link between them exists only in background knowledge: an engine needs fuel, a fuel tank holds fuel, an empty tank means there is no fuel, and no fuel makes an engine stop. A graph makes this chain explicit, and a GNN can learn to follow it."),

  H2("Building the Sub-Graph"),
  P("The sub-graph is built in four steps, following Lin et al. (2019) and Yasunaga et al. (2021)."),
  numbered("**Entity nodes.** Add one node for each grounded entity from Part B. Question entities are *engine* and *stop*; answer entities are *fuel tank* and *empty*.", "build"),
  numbered("**Bridge nodes.** Add knowledge-graph nodes that lie on short paths (one or two hops) between question and answer entities. Here these are *fuel* and *lack of fuel*.", "build"),
  numbered("**Typed edges.** Add every triplet that links two nodes in the set, keeping its relation type: HasSubevent, Uses, Contains, HasState, Implies, and Causes. Each edge is also added in the reverse direction with its own relation type, so messages can flow both ways.", "build"),
  numbered("**Context node.** Add one extra node, *z*, that stands for the text of the question and the choice. Connect *z* to every question and answer entity with a Context edge. This node is how the text enters the graph (see the final section).", "build"),
  P("Figure 2 shows the result. A separate sub-graph is built for each answer choice. For choice B (“The driver turned on the radio”), the only links between the radio and the engine run through weak PartOf edges (both are part of a car), and no Causes edge reaches *stop*."),
  ...figure(2, "Sub-Graph for “Why Did the Engine Stop?” and “The Fuel Tank Was Empty.”", "fig2_subgraph.png", 600, 324,
    "Blue nodes come from the question, green nodes from the answer choice, and white nodes are bridge concepts from the knowledge graph. Dashed lines connect the text context node *z* to each grounded entity. The red arrow marks the path the GNN should learn to follow. Figure created by the author."),

  H2("Walking the Path With a Graph Neural Network"),
  P("A GNN computes a vector for each node by passing messages along edges (Scarselli et al., 2009; Gilmer et al., 2017). After *L* rounds, each node’s vector holds information from nodes up to *L* hops away. Because our edges have types, we use a relational graph convolutional network (R-GCN; Schlichtkrull et al., 2018), which learns one weight matrix per relation type, and we add attention over incoming edges, as in graph attention networks (Veličković et al., 2018). This combination is close to the GNN used in QA-GNN (Yasunaga et al., 2021)."),
  H3("Node Initialization"),
  P("Each entity node *v* starts with an embedding of its name from a text encoder, *h*_{v}^{(0)} = *e*(name_{v}). The context node starts with the text vector *z*, defined in Equation 14 below."),
  H3("Message Passing"),
  P("At layer *l*, each node *v* collects messages from its neighbors *u*. A message is the neighbor’s vector multiplied by the weight matrix of the edge’s relation type *r*, and each message is weighted by an attention score α:"),
  EQ(9),
  EQ(10),
  PN("Here 𝒩_{v}^{r} is the set of neighbors linked to *v* by relation *r*, *e*_{r} is a learned embedding of the relation, *d* is the vector size, and the softmax runs over all edges coming into *v*. The relation matrices *W*_{r} let the model treat a Causes edge very differently from a PartOf edge. The attention α lets each node decide which neighbor matters most for this question."),
  H3("Walking the Path"),
  P("Each layer moves information one hop. The path from the answer to the question in Figure 2 has two hops (*empty* → *lack of fuel* → *stop*), plus one hop from *z*, so *L* = 3 layers are enough. Table 3 traces what happens, layer by layer, in a trained model."),
  ...apaTable(3, "How Information Moves Along the Path, Layer by Layer",
    ["Layer", "Edge followed", "What the receiving node learns"],
    [["1", "empty → lack of fuel (Implies)", "*lack of fuel* now carries the fact that the tank is empty."],
     ["1", "fuel tank → empty (HasState)", "*empty* is tied to the fuel tank, not to some other container."],
     ["2", "lack of fuel → stop (Causes)", "*stop* now carries “empty tank → no fuel → causes stop.”"],
     ["2", "engine → stop (HasSubevent)", "*stop* is tied to the engine, the subject of the question."],
     ["3", "stop → z (Context, reverse)", "*z* receives the full causal chain and can compare it with the question."]],
    [900, 3400, 5060],
    "In a trained model, attention α is high on the Causes and Implies edges for a “why” question and low on edges such as PartOf."),
  P("After training, the path can also be read back out, which makes the answer explainable. For a path π = (*v*_{0}, *v*_{1}, …, *v*_{k}) from an answer entity to a question entity, the path score is the sum of the log attention weights along it, averaged over layers:"),
  EQ(11),
  PN("The highest-scoring path is the explanation. For choice A it is *empty* → *lack of fuel* → *stop*, which matches the human reasoning. KagNet (Lin et al., 2019) and MHGRN (Feng et al., 2020) score paths in a similar way."),
  H3("Scoring and Training"),
  P("For each choice *a*, the model pools the node vectors with attention guided by the text, then scores the choice with a small feed-forward network:"),
  EQ(12),
  EQ(13),
  PN("The model is trained with cross-entropy, −log *P*(*a*^{∗} | *q*), where *a*^{∗} is the correct choice. Unlike the original MemNN, this needs no labels for the supporting facts. The model learns which edges to attend to from the answer labels alone, the same move that end-to-end memory networks made (Sukhbaatar et al., 2015)."),
  H3("Example Code"),
  P("The code below is a compact PyTorch version of Equations 9 to 13. The full script, `graph_reasoning_demo.py`, also builds the graph for each choice, runs one training step, and extracts the best path. It was run to check that it works end to end. Because the demo uses a stand-in text encoder and is not trained, its probabilities are close to one third each; it only shows the mechanics."),
  ...codeBlock([
    ...fileLines("graph_reasoning_demo.py", 52, 72),
    "",
    ...fileLines("graph_reasoning_demo.py", 75, 89),
  ]),

  H2("Combining the Question Text With the Extracted Relations"),
  P("The graph alone is not enough. It knows that empty tanks cause engines to stop, but it does not know which question was asked. The text alone is not enough either. A language model may not hold the fuel fact reliably, and it cannot show its reasoning. The model therefore fuses the two at four points."),
  numbered("**Encode the question and choice together.** The question and the choice are joined into one input and passed through a pre-trained encoder such as BERT (Devlin et al., 2019). The vector at the [CLS] position is the text representation:", "fuse"),
  EQ(14),
  PN("Joining *q* and *a* lets the encoder’s self-attention compare every word of the question with every word of the choice before the graph is used."),
  numbered("**Put the text inside the graph.** The vector *z* becomes the starting state of the context node, which is linked to every question and answer entity (Figure 2). During message passing, the text sends information into the graph and the graph sends the reasoning chain back into *z*. This is the “joint graph” design of QA-GNN (Yasunaga et al., 2021).", "fuse"),
  numbered("**Use the text to prune and weight the graph.** Knowledge graphs return many loosely related nodes. Each node *v* gets a relevance score from the encoder, ρ_{v} = *f*_{LM}([*q*; *a*; name_{v}]), and low-scoring nodes are dropped or down-weighted. The attention in Equation 10 is also driven by the text, since the query vector comes from nodes that already carry *z*.", "fuse"),
  numbered("**Combine at the output.** The final score in Equation 13 uses *z*, the updated context node *h*_{z}^{(L)}, and the pooled graph vector *g*. The text and the graph each add evidence, and the network learns how much to trust each.", "fuse"),
  P("The triplets enter through two routes. Their entities set the graph’s nodes, and their relations set the edge types, each with its own matrix *W*_{r} and embedding *e*_{r}. A simpler option is also possible: write each triplet out as a short sentence (“empty implies lack of fuel; lack of fuel causes stop”) and append it to the text input of Equation 14. This is easy to set up, but it loses the explicit graph structure and the readable path."),

  H2("Connecting Back to Memory Networks"),
  P("The two halves of this paper describe the same idea at two levels. In a memory network, the memory holds facts and the *O* module chains them hop by hop with Equations 2 and 3. In the graph approach, the extracted triplets *are* the memory, and each GNN layer is one hop. The difference is how the chaining happens. A MemNN must discover which memories belong together by learning word similarity. A GNN is told how facts connect through typed edges, so it can follow a Causes edge even when the two sentences share no words. This is why the graph approach handles “Why did the engine stop?” well, and why the plain word-matching *s*_{O} of Equation 5 would struggle with it. The memory network idea—keep facts in an explicit store and learn to read them in several hops—remains the foundation of both."),
];

const conclusion = [
  H1("Conclusion"),
  P("Memory networks give a learning model a store of facts it can write to and read from. The MemNN for text keeps each sentence in its own slot, scores memories with an embedding model, chains up to two facts per question, and learns with a margin ranking loss that needs labeled supporting facts. Relation extraction turns questions and answer choices into triplets that can serve as memories or as the edges of a graph. A relational GNN over a small sub-graph can then walk from “The fuel tank was empty” to “Why did the engine stop?” through *lack of fuel*, while a context node and a joint text encoder keep the reasoning tied to the actual question. The path the GNN follows also serves as an explanation of its answer."),
];

const references = [
  pageBreak(),
  H1("References"),
  REF("Banko, M., Cafarella, M. J., Soderland, S., Broadhead, M., & Etzioni, O. (2007). Open information extraction from the web. In *Proceedings of the 20th International Joint Conference on Artificial Intelligence* (pp. 2670–2676). IJCAI."),
  REF("Bordes, A., Chopra, S., & Weston, J. (2014). Question answering with subgraph embeddings. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing* (pp. 615–620). Association for Computational Linguistics. https://doi.org/10.3115/v1/D14-1067"),
  REF("Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423"),
  REF("Fader, A., Zettlemoyer, L., & Etzioni, O. (2013). Paraphrase-driven learning for open question answering. In *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics* (pp. 1608–1618). Association for Computational Linguistics."),
  REF("Feng, Y., Chen, X., Lin, B. Y., Wang, P., Yan, J., & Ren, X. (2020). Scalable multi-hop relational reasoning for knowledge-aware question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing* (pp. 1295–1309). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-main.99"),
  REF("Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1263–1272). PMLR."),
  REF("Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735"),
  REF("Huguet Cabot, P.-L., & Navigli, R. (2021). REBEL: Relation extraction by end-to-end language generation. In *Findings of the Association for Computational Linguistics: EMNLP 2021* (pp. 2370–2381). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.findings-emnlp.204"),
  REF("Kumar, A., Irsoy, O., Ondruska, P., Iyyer, M., Bradbury, J., Gulrajani, I., Zhong, V., Paulus, R., & Socher, R. (2016). Ask me anything: Dynamic memory networks for natural language processing. In *Proceedings of the 33rd International Conference on Machine Learning* (pp. 1378–1387). PMLR."),
  REF("Lin, B. Y., Chen, X., Chen, J., & Ren, X. (2019). KagNet: Knowledge-aware graph networks for commonsense reasoning. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 2829–2839). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1282"),
  REF("Mintz, M., Bills, S., Snow, R., & Jurafsky, D. (2009). Distant supervision for relation extraction without labeled data. In *Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th International Joint Conference on Natural Language Processing of the AFNLP* (pp. 1003–1011). Association for Computational Linguistics."),
  REF("Sap, M., Le Bras, R., Allaway, E., Bhagavatula, C., Lourie, N., Rashkin, H., Roof, B., Smith, N. A., & Choi, Y. (2019). ATOMIC: An atlas of machine commonsense for if-then reasoning. In *Proceedings of the AAAI Conference on Artificial Intelligence, 33*(1), 3027–3035. https://doi.org/10.1609/aaai.v33i01.33013027"),
  REF("Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2009). The graph neural network model. *IEEE Transactions on Neural Networks, 20*(1), 61–80. https://doi.org/10.1109/TNN.2008.2005605"),
  REF("Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). Modeling relational data with graph convolutional networks. In A. Gangemi et al. (Eds.), *The semantic web: ESWC 2018* (pp. 593–607). Springer. https://doi.org/10.1007/978-3-319-93417-4_38"),
  REF("Speer, R., Chin, J., & Havasi, C. (2017). ConceptNet 5.5: An open multilingual graph of general knowledge. In *Proceedings of the AAAI Conference on Artificial Intelligence, 31*(1), 4444–4451. https://doi.org/10.1609/aaai.v31i1.11164"),
  REF("Sukhbaatar, S., Szlam, A., Weston, J., & Fergus, R. (2015). End-to-end memory networks. In *Advances in Neural Information Processing Systems 28* (pp. 2440–2448). Curran Associates."),
  REF("Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In *Proceedings of the 6th International Conference on Learning Representations*. https://arxiv.org/abs/1710.10903"),
  REF("Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling up to large vocabulary image annotation. In *Proceedings of the 22nd International Joint Conference on Artificial Intelligence* (pp. 2764–2770). AAAI Press."),
  REF("Weston, J., Bordes, A., Chopra, S., Rush, A. M., van Merriënboer, B., Joulin, A., & Mikolov, T. (2016). Towards AI-complete question answering: A set of prerequisite toy tasks. In *Proceedings of the 4th International Conference on Learning Representations*. https://arxiv.org/abs/1502.05698"),
  REF("Weston, J., Chopra, S., & Bordes, A. (2015). Memory networks. In *Proceedings of the 3rd International Conference on Learning Representations*. https://arxiv.org/abs/1410.3916"),
  REF("Yasunaga, M., Ren, H., Bosselut, A., Liang, P., & Leskovec, J. (2021). QA-GNN: Reasoning with language models and knowledge graphs for question answering. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 535–546). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.naacl-main.45"),
];

// =====================================================================
const numCfg = (ref, fmt, text) => ({
  reference: ref,
  levels: [{ level: 0, format: fmt, text, alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
});

const doc = new Document({
  creator: "Student",
  title: "Memory Networks for Question Answering",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "000000" },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0, line: DOUBLE }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "000000" },
        paragraph: { spacing: { before: 0, after: 0, line: DOUBLE }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true, color: "000000" },
        paragraph: { spacing: { before: 0, after: 0, line: DOUBLE }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      numCfg("bullets", LevelFormat.BULLET, "•"),
      numCfg("flow", LevelFormat.DECIMAL, "%1."),
      numCfg("build", LevelFormat.DECIMAL, "%1."),
      numCfg("fuse", LevelFormat.DECIMAL, "%1."),
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
    children: [...titlePage, ...abstract, ...intro, ...partA, ...partB, ...partC, ...conclusion, ...references],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Assignment4_Memory_Networks_QA.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
