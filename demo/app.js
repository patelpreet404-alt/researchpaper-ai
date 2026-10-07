const answers = [
  {
    matches: ["rag", "retrieval augmented", "retrieval-augmented"],
    answer: "Retrieval-augmented generation (RAG) combines a retrieval system with a language model. It fetches relevant text from an external source, such as a PDF, and includes that text in the prompt so the response is grounded in the document.",
    page: 1,
    source: "Introduction to Retrieval-Augmented Generation",
    excerpt: "Relevant text from a source document is placed alongside a question so answers can draw on that evidence."
  },
  {
    matches: ["semantic", "search", "similar"],
    answer: "Semantic search looks for passages with similar meaning, even when they do not use the same words. The document describes converting text chunks and the query into vectors, then finding nearby vectors.",
    page: 3,
    source: "Semantic Search and FAISS",
    excerpt: "Semantic search represents passages and questions as vectors, then compares how close their meanings are."
  },
  {
    matches: ["faiss", "vector", "embedding", "nearest"],
    answer: "FAISS indexes vector embeddings. Given a new query vector, it finds the nearest neighbors: the chunks most likely to contain a relevant answer.",
    page: 3,
    source: "Semantic Search and FAISS",
    excerpt: "FAISS indexes embeddings and finds the nearest passages for a new query vector."
  },
  {
    matches: ["process", "pipeline", "chunk", "document", "upload", "app"],
    answer: "The sample guide describes a pipeline that chunks document text, creates embeddings, stores the vectors in FAISS, and retrieves relevant passages when a question is asked.",
    page: 2,
    source: "How This Sample Document Works",
    excerpt: "The pipeline splits document text into chunks, creates embeddings, and indexes them for retrieval."
  }
];

const messages = document.getElementById("messages");
const questionForm = document.getElementById("question-form");
const questionInput = document.getElementById("question-input");
const themeToggle = document.getElementById("theme-toggle");
const resetButton = document.getElementById("reset-button");
const welcomeTemplate = document.getElementById("empty-chat").cloneNode(true);
const paperSheet = document.getElementById("paper-sheet");
const composerButton = questionForm.querySelector('button[type="submit"]');
const pages = [
  { title: "Introduction to Retrieval-Augmented Generation", paragraphs: ["Retrieval-augmented generation combines a retrieval system with a language model. Relevant text from a source document is placed alongside a question so answers can draw on that evidence.", "Instead of relying only on a model’s learned knowledge, RAG looks up useful passages when a question arrives."] },
  { title: "How This Sample Document Works", paragraphs: ["The pipeline splits document text into chunks, creates embeddings, and indexes them for retrieval.", "When a question arrives, relevant chunks are retrieved and passed along as context for a grounded answer."] },
  { title: "Semantic Search and FAISS", paragraphs: ["Semantic search represents passages and questions as vectors, then compares how close their meanings are.", "FAISS indexes embeddings and finds the nearest passages for a new query vector."] }
];
let pendingAnswer = null;

function showPage(page) {
  const selected = pages[page - 1];
  if (!selected) return;
  paperSheet.replaceChildren();
  const number = document.createElement("span");
  number.className = "paper-sheet-page";
  number.textContent = `PAGE ${String(page).padStart(2, "0")} / 03`;
  const heading = document.createElement("h3");
  heading.textContent = selected.title;
  paperSheet.append(number, heading);
  selected.paragraphs.forEach((paragraph, index) => {
    if (index) {
      const rule = document.createElement("div");
      rule.className = "paper-rule";
      paperSheet.append(rule);
    }
    const text = document.createElement("p");
    text.textContent = paragraph;
    paperSheet.append(text);
  });
  document.querySelectorAll("[data-page]").forEach((button) => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.page) === page));
  });
}

function matchAnswer(question) {
  const normalized = question.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
  return answers.find((item) => item.matches.some((match) => normalized.includes(match))) || null;
}

function appendMessage(role, content, citation) {
  const row = document.createElement("div");
  row.className = `message message-${role}`;

  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "YOU" : "SAMPLE ANSWER";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = content;

  row.append(label, bubble);

  if (citation) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "citation";
    card.innerHTML = `<span class="citation-top"><span class="citation-icon">▤</span><strong>Sample RAG Guide</strong><span class="citation-page">Page ${citation.page}</span></span><span class="citation-excerpt"></span><span class="citation-action">View source page →</span>`;
    card.querySelector(".citation-excerpt").textContent = `“${citation.excerpt}”`;
    card.addEventListener("click", () => {
      showPage(citation.page);
      document.querySelector(".paper-preview").scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    row.append(card);
  }

  messages.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "end" });
}

function ask(question) {
  const trimmed = question.trim();
  if (!trimmed || pendingAnswer) return;

  document.getElementById("empty-chat")?.remove();
  appendMessage("user", trimmed);

  const loading = document.createElement("div");
  loading.className = "answer-loading";
  loading.setAttribute("role", "status");
  loading.textContent = "Finding a prepared answer in the sample guide…";
  messages.append(loading);
  composerButton.disabled = true;
  pendingAnswer = window.setTimeout(() => {
    loading.remove();
    const found = matchAnswer(trimmed);
    if (found) appendMessage("assistant", found.answer, found);
    else appendMessage("assistant", "This sample only contains prepared answers about RAG, semantic search, FAISS, and the document pipeline. Try one of the suggested questions to see a cited answer.");
    pendingAnswer = null;
    composerButton.disabled = false;
  }, 500);
  questionInput.value = "";
}

questionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(questionInput.value);
});

document.addEventListener("click", (event) => {
  const pageButton = event.target.closest("[data-page]");
  if (pageButton) showPage(Number(pageButton.dataset.page));
  const button = event.target.closest("[data-question]");
  if (button) ask(button.dataset.question);
});

resetButton.addEventListener("click", () => {
  if (pendingAnswer) window.clearTimeout(pendingAnswer);
  pendingAnswer = null;
  composerButton.disabled = false;
  messages.replaceChildren(welcomeTemplate.cloneNode(true));
  showPage(1);
  questionInput.focus();
});

try {
  const savedTheme = localStorage.getItem("researchpaper-demo-theme");
  if (savedTheme === "light" || savedTheme === "dark") {
    document.documentElement.dataset.theme = savedTheme;
  }
} catch {}

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("researchpaper-demo-theme", next);
  } catch {}
});
