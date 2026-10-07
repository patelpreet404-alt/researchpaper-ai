const answers = [
  {
    matches: ["rag", "retrieval augmented", "retrieval-augmented"],
    answer: "Retrieval-augmented generation (RAG) combines a retrieval system with a language model. It fetches relevant text from an external source, such as a PDF, and includes that text in the prompt so the response is grounded in the document.",
    page: 1,
    source: "Introduction to Retrieval-Augmented Generation"
  },
  {
    matches: ["semantic", "search", "similar"],
    answer: "Semantic search looks for passages with similar meaning, even when they do not use the same words. The document describes converting text chunks and the query into vectors, then finding nearby vectors.",
    page: 3,
    source: "Semantic Search and FAISS"
  },
  {
    matches: ["faiss", "vector", "embedding", "nearest"],
    answer: "FAISS indexes vector embeddings. Given a new query vector, it finds the nearest neighbors: the chunks most likely to contain a relevant answer.",
    page: 3,
    source: "Semantic Search and FAISS"
  },
  {
    matches: ["process", "pipeline", "chunk", "document", "upload", "app"],
    answer: "The sample guide describes a pipeline that chunks document text, creates embeddings, stores the vectors in FAISS, and retrieves relevant passages when a question is asked.",
    page: 2,
    source: "How This Sample Document Works"
  }
];

const messages = document.getElementById("messages");
const questionForm = document.getElementById("question-form");
const questionInput = document.getElementById("question-input");
const themeToggle = document.getElementById("theme-toggle");
const resetButton = document.getElementById("reset-button");
const welcomeTemplate = document.getElementById("empty-chat").cloneNode(true);

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
    const link = document.createElement("a");
    link.className = "citation";
    link.href = `/sample-rag-guide.pdf#page=${citation.page}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `↗ Sample RAG Guide · page ${citation.page} · ${citation.source}`;
    row.append(link);
  }

  messages.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "end" });
}

function ask(question) {
  const trimmed = question.trim();
  if (!trimmed) return;

  document.getElementById("empty-chat")?.remove();
  appendMessage("user", trimmed);

  const found = matchAnswer(trimmed);
  if (found) {
    appendMessage("assistant", found.answer, found);
  } else {
    appendMessage("assistant", "This sample only contains prepared answers about RAG, semantic search, FAISS, and the document pipeline. Try one of the suggested questions to see a cited answer.");
  }

  questionInput.value = "";
  questionInput.focus();
}

questionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(questionInput.value);
});

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-question]");
  if (button) ask(button.dataset.question);
});

resetButton.addEventListener("click", () => {
  messages.replaceChildren(welcomeTemplate.cloneNode(true));
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
