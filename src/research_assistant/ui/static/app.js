const form = document.querySelector("#ask");
const question = document.querySelector("#question");
const runButton = document.querySelector("#run");
const status = document.querySelector("#status");
const errorBox = document.querySelector("#error");
const report = document.querySelector("#report");
const papers = document.querySelector("#papers");

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(question.value);
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

loadPapers();

async function loadPapers() {
  try {
    const response = await fetch("/api/corpus");
    const payload = await response.json();
    if (!response.ok) {
      papers.append(item(payload.error || "The corpus could not be read."));
      return;
    }
    document.querySelector("#corpus-title").textContent = `Papers in this run (${payload.papers.length})`;
    payload.papers.forEach((paper) => {
      const row = document.createElement("li");
      row.textContent = paper.title;
      const id = document.createElement("span");
      id.textContent = paper.id;
      row.append(id);
      papers.append(row);
    });
  } catch (error) {
    papers.append(item("The corpus could not be read."));
  }
}

async function ask(text) {
  hide(errorBox);
  report.hidden = true;
  report.replaceChildren();
  runButton.disabled = true;
  status.hidden = false;
  status.textContent = "Searching the papers, then checking claims. The first summary and the contradiction check run together.";
  try {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      showError(payload.error || "The research run failed.");
      return;
    }
    renderReport(payload);
  } catch (error) {
    showError("The page lost contact with the local server.");
  } finally {
    runButton.disabled = false;
    status.hidden = true;
  }
}

function renderReport(payload) {
  report.hidden = false;
  report.append(heading("Overview"), paragraph(payload.overview, "overview"));

  const facts = document.createElement("p");
  facts.className = "meta";
  const count = payload.passage_count === 1 ? "1 passage" : `${payload.passage_count} passages`;
  facts.textContent = `Retrieved ${count}.`;
  report.append(facts);

  report.append(heading("Themes"));
  if (!payload.themes.length) {
    report.append(paragraph("No supported claims remained after fact-checking."));
  }
  payload.themes.forEach((theme) => {
    const section = document.createElement("section");
    section.className = "theme";
    const title = document.createElement("h3");
    title.textContent = theme.theme;
    section.append(title);
    theme.claims.forEach((claim) => section.append(claimBlock(claim)));
    report.append(section);
  });

  report.append(heading("Contradictions"));
  if (!payload.contradictions.length) {
    report.append(paragraph(payload.contradiction_note || "No contradiction was found in the retrieved passages."));
  }
  payload.contradictions.forEach((item) => report.append(dispute(item)));

  report.append(heading("Dropped claims"));
  if (!payload.dropped.length) {
    report.append(paragraph("None."));
  } else {
    const list = document.createElement("ul");
    list.className = "dropped";
    payload.dropped.forEach((item) => {
      const row = document.createElement("li");
      row.textContent = `${item.text} — ${item.reason}`;
      list.append(row);
    });
    report.append(list);
  }

  if (payload.sub_queries && payload.sub_queries.length) {
    const queries = document.createElement("p");
    queries.className = "meta";
    queries.textContent = `Sub-queries: ${payload.sub_queries.join(" · ")}`;
    report.append(queries);
  }

  const saved = document.createElement("p");
  saved.className = "saved";
  saved.textContent = `Saved ${payload.markdown_path} and ${payload.json_path}.`;
  report.append(saved);
  report.scrollIntoView({ behavior: "smooth", block: "start" });
}

function claimBlock(claim) {
  const block = document.createElement("div");
  block.className = "claim";
  block.append(paragraph(claim.text));
  if (claim.quote) block.append(quote(claim.quote));
  claim.sources.forEach((source) => block.append(sourceLine(source)));
  return block;
}

function dispute(item) {
  const section = document.createElement("section");
  section.className = "dispute";
  const title = document.createElement("h3");
  title.textContent = item.topic;
  const sides = document.createElement("div");
  sides.className = "sides";
  sides.append(position("Position A", item.position_a, item.quote_a, item.sources_a));
  sides.append(position("Position B", item.position_b, item.quote_b, item.sources_b));
  section.append(title, sides);
  return section;
}

function position(label, text, quoted, sources) {
  const block = document.createElement("div");
  block.className = "position";
  const name = document.createElement("p");
  name.className = "meta";
  name.textContent = label;
  block.append(name, paragraph(text));
  if (quoted) block.append(quote(quoted));
  sources.forEach((source) => block.append(sourceLine(source)));
  return block;
}

function sourceLine(source) {
  const line = document.createElement("p");
  line.className = "source";
  line.textContent = `${source.title} (${source.passage_id})`;
  return line;
}

function heading(text) {
  const node = document.createElement("h2");
  node.textContent = text;
  return node;
}

function paragraph(text, className) {
  const node = document.createElement("p");
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

function quote(text) {
  const node = document.createElement("blockquote");
  node.textContent = text;
  return node;
}

function item(text) {
  const node = document.createElement("li");
  node.textContent = text;
  return node;
}

function showError(message) {
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function hide(node) {
  node.hidden = true;
  node.textContent = "";
}
