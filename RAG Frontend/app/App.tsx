import { useState, useRef, useEffect, useCallback } from "react";
import {
  Moon,
  Sun,
  Send,
  Bot,
  User,
  Plus,
  Search,
  MessageSquareText,
  Clock3,
  PanelLeft,
  FileText,
  ChevronDown,
  ChevronUp,
  Copy,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  Check,
  Sparkles,
  X,
  RotateCcw,
  Trash2,
} from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "./components/ui/alert-dialog";

// ─── Types ─────────────────────────────────────────────────────────────────

type Role = "user" | "assistant";

interface Source {
  id: string;
  title: string;
  excerpt: string;
  type: "pdf" | "doc" | "web" | "txt";
}

interface Feedback {
  vote: "up" | "down" | null;
}

interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: Date;
  sources?: Source[];
  feedback?: Feedback;
  isStreaming?: boolean;
}

interface ChatThread {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

// ─── Demo data ──────────────────────────────────────────────────────────────

const SUGGESTED = [
  "What's your return policy?",
  "How do I track my order?",
  "Do you offer enterprise pricing?",
  "What payment methods are accepted?",
  "How long does shipping take?",
];

const DEMO_SOURCES: Source[] = [
  { id: "s1", title: "Returns & Refunds Policy", excerpt: "Items can be returned within 30 days of delivery in original condition...", type: "pdf" },
  { id: "s2", title: "Shipping Information Guide", excerpt: "Standard shipping takes 5-7 business days. Express options are available...", type: "doc" },
  { id: "s3", title: "Customer FAQ — Order Tracking", excerpt: "You can track your order using the tracking number sent via email...", type: "web" },
];

const DEMO_RESPONSES: Record<string, string> = {
  "What's your return policy?":
    "We offer a **30-day hassle-free return policy** for all products. Items must be in their original condition with all packaging intact.\n\nHere's how to initiate a return:\n1. Log in to your account and navigate to **Orders**.\n2. Select the order and click **Request Return**.\n3. Choose a reason and print the prepaid shipping label.\n4. Drop it off at any authorized carrier location.\n\nRefunds are processed within **3–5 business days** after we receive the item.",
  "How do I track my order?":
    "Tracking your order is easy! As soon as your package ships, you'll receive a confirmation email with a **tracking number** and a direct link.\n\nYou can also:\n- Visit **My Orders** in your account dashboard.\n- Enter your order number on our [Track Order](#) page.\n- Use the carrier's website directly (UPS, FedEx, or USPS).\n\nIf your tracking hasn't updated in 48 hours, please contact our support team.",
  "Do you offer enterprise pricing?":
    "Yes — we have a dedicated **Enterprise plan** built for teams of 25 or more. It includes:\n\n- **Volume discounts** starting at 20% off\n- Dedicated account manager\n- Priority 24/7 support\n- Custom SLA and compliance documentation\n- SSO and advanced admin controls\n\nReach out to our sales team at **enterprise@company.com** or book a 30-minute demo call and we'll build a custom quote for you.",
  "What payment methods are accepted?":
    "We accept the following payment methods:\n\n**Cards:** Visa, Mastercard, American Express, Discover\n**Digital wallets:** Apple Pay, Google Pay, PayPal\n**Other:** Bank transfer (ACH), Purchase orders (enterprise accounts)\n\nAll transactions are secured with **TLS 1.3 encryption** and processed through Stripe. We never store raw card data.",
  "How long does shipping take?":
    "Delivery times depend on the shipping method you select at checkout:\n\n| Method | Estimated Delivery |\n|--------|-------------------|\n| Standard | 5–7 business days |\n| Expedited | 2–3 business days |\n| Overnight | Next business day |\n\nOrders placed before **2 PM EST** ship same day. Free standard shipping is available on all orders over $50.",
};

const DEFAULT_RESPONSE =
  "Thanks for reaching out! I searched our knowledge base and found some relevant information. Based on the retrieved documents, I can help you with that. Could you provide a bit more context so I can give you the most accurate answer?";

const INITIAL_MESSAGES: Message[] = [
  {
    id: generateId(),
    role: "assistant",
    content: "Hello! 👋 I'm your AI support assistant.\nHow can I help you today?",
    timestamp: new Date(Date.now() - 60 * 60 * 1000),
    sources: DEMO_SOURCES,
    feedback: { vote: null },
  },
  {
    id: generateId(),
    role: "user",
    content: "How do I reset my password?",
    timestamp: new Date(Date.now() - 58 * 60 * 1000),
  },
  {
    id: generateId(),
    role: "assistant",
    content:
      "To reset your password, follow these steps:\n\n1. Go to the login page.\n2. Click on \"Forgot Password?\"\n3. Enter your registered email address.\n4. Check your email for the reset link.\n5. Click the link and create a new password.\n\nIf you don't receive the email, please check your spam folder or contact our support team.",
    timestamp: new Date(Date.now() - 57 * 60 * 1000),
    sources: DEMO_SOURCES,
    feedback: { vote: null },
  },
];

const INITIAL_THREADS: ChatThread[] = [
  {
    id: "thread-1",
    title: "Password Reset",
    messages: INITIAL_MESSAGES,
    updatedAt: Date.now() - 57 * 60 * 1000,
  },
  {
    id: "thread-2",
    title: "Refund Policy",
    messages: [
      {
        id: generateId(),
        role: "user",
        content: "What is your refund policy?",
        timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000 - 20 * 60 * 1000),
      },
      {
        id: generateId(),
        role: "assistant",
        content: "We offer a 30-day refund window for eligible purchases.",
        timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000 - 19 * 60 * 1000),
        sources: DEMO_SOURCES,
        feedback: { vote: null },
      },
    ],
    updatedAt: Date.now() - 24 * 60 * 60 * 1000 - 19 * 60 * 1000,
  },
  {
    id: "thread-3",
    title: "Shipping Time",
    messages: [
      {
        id: generateId(),
        role: "user",
        content: "How long does shipping take?",
        timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      },
    ],
    updatedAt: Date.now() - 2 * 24 * 60 * 60 * 1000,
  },
];

const INITIAL_EMPTY_THREAD = {
  id: generateId(),
  title: "New Chat",
  messages: [],
  updatedAt: Date.now(),
};

function groupThreads(threads: ChatThread[]) {
  return {
    "Chat History": threads.slice().sort((a, b) => b.updatedAt - a.updatedAt),
  };
}

function generateId() {
  return Math.random().toString(36).slice(2, 9);
}

function formatTime(d: Date) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ─── Markdown-lite renderer ─────────────────────────────────────────────────

function renderMarkdown(text: string) {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let listItems: string[] = [];
  let tableRows: string[][] = [];
  let inTable = false;
  let key = 0;

  const flushList = () => {
    if (listItems.length) {
      elements.push(
        <ol key={key++} className="list-decimal pl-5 space-y-1 my-2 text-sm leading-relaxed">
          {listItems.map((li, i) => (
            <li key={i}>{renderInline(li)}</li>
          ))}
        </ol>
      );
      listItems = [];
    }
  };

  const flushTable = () => {
    if (tableRows.length) {
      const [header, , ...body] = tableRows;
      elements.push(
        <div key={key++} className="overflow-x-auto my-3">
          <table className="min-w-full text-sm border-collapse">
            <thead>
              <tr>
                {header.map((cell, i) => (
                  <th key={i} className="text-left px-3 py-1.5 font-semibold bg-muted border border-border text-xs uppercase tracking-wide text-muted-foreground">
                    {cell.trim()}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {body.map((row, ri) => (
                <tr key={ri} className="even:bg-muted/40">
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-3 py-1.5 border border-border">
                      {cell.trim()}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableRows = [];
      inTable = false;
    }
  };

  lines.forEach((line) => {
    if (line.startsWith("|")) {
      inTable = true;
      tableRows.push(line.split("|").filter((_, i, a) => i > 0 && i < a.length - 1));
      return;
    }
    if (inTable) flushTable();

    const olMatch = line.match(/^(\d+)\.\s+(.*)/);
    if (olMatch) {
      flushList();
      listItems.push(olMatch[2]);
      return;
    }
    flushList();

    if (line.startsWith("## ")) {
      elements.push(<h3 key={key++} className="font-semibold text-base mt-4 mb-1">{line.slice(3)}</h3>);
    } else if (line.startsWith("# ")) {
      elements.push(<h2 key={key++} className="font-bold text-lg mt-4 mb-1">{line.slice(2)}</h2>);
    } else if (line.trim() === "") {
      elements.push(<div key={key++} className="h-2" />);
    } else {
      elements.push(<p key={key++} className="text-sm leading-relaxed">{renderInline(line)}</p>);
    }
  });

  flushList();
  if (inTable) flushTable();

  return elements;
}

function renderInline(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} className="font-semibold">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={i} className="bg-muted text-primary font-mono text-xs px-1 py-0.5 rounded">{part.slice(1, -1)}</code>;
    }
    const linkMatch = part.match(/\[([^\]]+)\]\(([^)]+)\)/);
    if (linkMatch) {
      return <a key={i} href={linkMatch[2]} className="text-primary underline underline-offset-2 hover:text-primary/80">{linkMatch[1]}</a>;
    }
    return part;
  });
}

// ─── Source icon ────────────────────────────────────────────────────────────

function SourceIcon({ type }: { type: Source["type"] }) {
  const base = "w-7 h-7 rounded-md flex items-center justify-center text-[10px] font-bold shrink-0";
  const map: Record<Source["type"], [string, string]> = {
    pdf: ["bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400", "PDF"],
    doc: ["bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400", "DOC"],
    web: ["bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400", "WEB"],
    txt: ["bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400", "TXT"],
  };
  const [cls, label] = map[type];
  return <span className={`${base} ${cls}`}>{label}</span>;
}

// ─── Sources panel ──────────────────────────────────────────────────────────

function SourcesPanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3 border border-border rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs font-medium text-muted-foreground hover:bg-muted/60 transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <FileText className="w-3.5 h-3.5" />
          {sources.length} source{sources.length !== 1 ? "s" : ""} retrieved
        </span>
        {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      {open && (
        <div className="border-t border-border divide-y divide-border">
          {sources.map((src) => (
            <div key={src.id} className="flex items-start gap-3 px-3.5 py-3 hover:bg-muted/40 transition-colors">
              <SourceIcon type={src.type} />
              <div className="min-w-0">
                <p className="text-xs font-semibold text-foreground leading-snug">{src.title}</p>
                <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed line-clamp-2">{src.excerpt}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Message actions ─────────────────────────────────────────────────────────

function MessageActions({
  content,
  feedback,
  onFeedback,
  onRegenerate,
}: {
  content: string;
  feedback: Feedback;
  onFeedback: (v: "up" | "down") => void;
  onRegenerate: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex items-center gap-1 mt-2.5">
      <button
        onClick={copy}
        title="Copy"
        className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
      >
        {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
        <span>{copied ? "Copied" : "Copy"}</span>
      </button>
      <button
        onClick={onRegenerate}
        title="Regenerate"
        className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
      >
        <RefreshCw className="w-3.5 h-3.5" />
        <span>Regenerate</span>
      </button>
      <div className="flex items-center gap-0.5 ml-auto">
        <button
          onClick={() => onFeedback("up")}
          title="Helpful"
          className={`p-1.5 rounded-lg transition-colors ${
            feedback.vote === "up"
              ? "text-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
          }`}
        >
          <ThumbsUp className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => onFeedback("down")}
          title="Not helpful"
          className={`p-1.5 rounded-lg transition-colors ${
            feedback.vote === "down"
              ? "text-red-500 bg-red-50 dark:bg-red-900/20"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
          }`}
        >
          <ThumbsDown className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}

// ─── Typing indicator ────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex items-end gap-3 max-w-2xl">
      <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center shrink-0 shadow-sm">
        <Bot className="w-4 h-4 text-white" />
      </div>
      <div className="bg-card border border-border rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1.5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="w-2 h-2 rounded-full bg-muted-foreground/60 animate-bounce"
              style={{ animationDelay: `${i * 150}ms`, animationDuration: "900ms" }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Single message bubble ───────────────────────────────────────────────────

function MessageBubble({
  message,
  onFeedback,
  onRegenerate,
}: {
  message: Message;
  onFeedback: (id: string, v: "up" | "down") => void;
  onRegenerate: (id: string) => void;
}) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex items-end justify-end gap-3">
        <div className="max-w-[80%] md:max-w-[65%]">
          <div className="bg-primary text-primary-foreground rounded-2xl rounded-br-sm px-4 py-3 shadow-sm">
            <p className="text-sm leading-relaxed">{message.content}</p>
          </div>
          <p className="text-[11px] text-muted-foreground mt-1 text-right pr-0.5">{formatTime(message.timestamp)}</p>
        </div>
        <div className="w-8 h-8 rounded-full bg-muted border border-border flex items-center justify-center shrink-0 shadow-sm">
          <User className="w-4 h-4 text-muted-foreground" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 max-w-2xl">
      <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center shrink-0 shadow-sm mt-0.5">
        <Bot className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
          <div className="prose-sm text-foreground space-y-0.5">
            {renderMarkdown(message.content)}
          </div>
          {message.sources && message.sources.length > 0 && (
            <SourcesPanel sources={message.sources} />
          )}
        </div>
        <MessageActions
          content={message.content}
          feedback={message.feedback ?? { vote: null }}
          onFeedback={(v) => onFeedback(message.id, v)}
          onRegenerate={() => onRegenerate(message.id)}
        />
        <p className="text-[11px] text-muted-foreground mt-0.5 pl-0.5">{formatTime(message.timestamp)}</p>
      </div>
    </div>
  );
}

// ─── Empty state ─────────────────────────────────────────────────────────────

function EmptyState({ onSuggest }: { onSuggest: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4 py-16">
      <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-5">
        <Sparkles className="w-7 h-7 text-primary" />
      </div>
      <h2 className="text-xl font-semibold text-foreground mb-2">How can I help you today?</h2>
      <p className="text-sm text-muted-foreground max-w-sm leading-relaxed mb-8">
        I have access to your product documentation, policies, and support articles. Ask me anything.
      </p>
      <div className="flex flex-wrap justify-center gap-2 max-w-lg">
        {SUGGESTED.map((q) => (
          <button
            key={q}
            onClick={() => onSuggest(q)}
            className="px-3.5 py-2 rounded-full border border-border bg-card text-sm text-foreground hover:bg-muted hover:border-primary/30 hover:text-primary transition-all duration-150 shadow-sm"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────

export default function App() {
  const [dark, setDark] = useState(false);
  const [threads, setThreads] = useState<ChatThread[]>(() => [INITIAL_EMPTY_THREAD, ...INITIAL_THREADS]);
  const [activeThreadId, setActiveThreadId] = useState(INITIAL_EMPTY_THREAD.id);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [threadToDelete, setThreadToDelete] = useState<ChatThread | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  const activeThread = threads.find((thread) => thread.id === activeThreadId) ?? threads[0];
  const messages = activeThread?.messages ?? [];
  const normalizedQuery = searchQuery.trim().toLowerCase();
  const filteredThreads = normalizedQuery
    ? threads.filter((thread) => {
        const titleMatches = thread.title.toLowerCase().includes(normalizedQuery);
        const messageMatches = thread.messages.some((message) =>
          message.content.toLowerCase().includes(normalizedQuery)
        );
        return titleMatches || messageMatches;
      })
    : threads;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const autoResize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;

      const userMsg: Message = {
        id: generateId(),
        role: "user",
        content: trimmed,
        timestamp: new Date(),
      };

      setThreads((prev) =>
        prev.map((thread) =>
          thread.id === activeThreadId
            ? {
                ...thread,
                title: thread.messages.length === 0 ? trimmed.slice(0, 24) : thread.title,
                messages: [...thread.messages, userMsg],
                updatedAt: Date.now(),
              }
            : thread
        )
      );
      setInput("");
      setLoading(true);
      setShowSuggestions(false);
      if (textareaRef.current) textareaRef.current.style.height = "auto";

      await new Promise((r) => setTimeout(r, 1200 + Math.random() * 600));

      const responseText =
        DEMO_RESPONSES[trimmed] ?? DEFAULT_RESPONSE;

      const aiMsg: Message = {
        id: generateId(),
        role: "assistant",
        content: responseText,
        timestamp: new Date(),
        sources: DEMO_SOURCES,
        feedback: { vote: null },
      };

      setThreads((prev) =>
        prev.map((thread) =>
          thread.id === activeThreadId
            ? {
                ...thread,
                messages: [...thread.messages, aiMsg],
                updatedAt: Date.now(),
              }
            : thread
        )
      );
      setLoading(false);
    },
    [activeThreadId, loading]
  );

  const handleFeedback = (id: string, vote: "up" | "down") => {
    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThreadId
          ? {
              ...thread,
              messages: thread.messages.map((m) =>
                m.id === id
                  ? { ...m, feedback: { vote: m.feedback?.vote === vote ? null : vote } }
                  : m
              ),
              updatedAt: Date.now(),
            }
          : thread
      )
    );
  };

  const handleRegenerate = async (id: string) => {
    const msg = messages.find((m) => m.id === id);
    if (!msg || loading) return;
    const prevUser = [...messages].reverse().find((m, i, arr) => {
      const idx = arr.indexOf(m);
      return m.role === "user" && idx > arr.indexOf(msg);
    });
    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThreadId
          ? { ...thread, messages: thread.messages.filter((m) => m.id !== id), updatedAt: Date.now() }
          : thread
      )
    );
    setLoading(true);
    await new Promise((r) => setTimeout(r, 1000));
    const aiMsg: Message = {
      id: generateId(),
      role: "assistant",
      content: prevUser ? (DEMO_RESPONSES[prevUser.content] ?? DEFAULT_RESPONSE) : DEFAULT_RESPONSE,
      timestamp: new Date(),
      sources: DEMO_SOURCES,
      feedback: { vote: null },
    };
    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThreadId
          ? { ...thread, messages: [...thread.messages, aiMsg], updatedAt: Date.now() }
          : thread
      )
    );
    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const clearChat = () => {
    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThreadId ? { ...thread, messages: [], updatedAt: Date.now() } : thread
      )
    );
    setShowSuggestions(true);
  };

  const createNewChat = () => {
    const newThread: ChatThread = {
      id: generateId(),
      title: "New Chat",
      messages: [],
      updatedAt: Date.now(),
    };
    setThreads((prev) => [newThread, ...prev]);
    setActiveThreadId(newThread.id);
    setInput("");
    setShowSuggestions(true);
  };

  const deleteChat = (threadId: string) => {
    setThreads((prev) => {
      const remaining = prev.filter((thread) => thread.id !== threadId);

      if (threadId === activeThreadId) {
        if (remaining.length > 0) {
          setActiveThreadId(remaining[0].id);
          setShowSuggestions(true);
        } else {
          const fallbackThread: ChatThread = {
            id: generateId(),
            title: "New Chat",
            messages: [],
            updatedAt: Date.now(),
          };
          setActiveThreadId(fallbackThread.id);
          setShowSuggestions(true);
          return [fallbackThread];
        }
      }

      return remaining;
    });

    setInput("");
  };

  const confirmDeleteChat = () => {
    if (!threadToDelete) return;
    deleteChat(threadToDelete.id);
    setThreadToDelete(null);
  };

  const groupedThreads = groupThreads(filteredThreads);

  const hasMessages = messages.length > 0;

  return (
    <div className="h-screen flex bg-background overflow-hidden" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
      {mobileSidebarOpen && (
        <button
          type="button"
          aria-label="Close sidebar overlay"
          onClick={() => setMobileSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-black/30 md:hidden"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 md:z-auto flex shrink-0 flex-col border-r border-border bg-card/70 backdrop-blur-sm transition-all duration-300 ease-out overflow-hidden md:overflow-hidden md:translate-x-0 ${
          mobileSidebarOpen ? "translate-x-0 w-[min(22rem,90vw)]" : "-translate-x-full w-[min(22rem,90vw)] md:translate-x-0"
        } ${sidebarOpen ? "md:w-80" : "md:w-0 md:border-r-0"}`}
      >
        <div className="p-3 sm:p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <button
              onClick={createNewChat}
              className="flex-1 inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-3 sm:px-4 py-2.5 sm:py-3 text-sm font-medium text-primary-foreground shadow-sm hover:bg-primary/90 transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Chat
            </button>
          </div>

          <div className="mt-3 relative">
            <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search chats..."
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="w-full rounded-xl border border-border bg-background pl-9 pr-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-primary/20"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 inline-flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                aria-label="Clear search"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-4 sm:space-y-5">
          {Object.entries(groupedThreads).length > 0 ? (
            Object.entries(groupedThreads).map(([groupName, groupThreads]) => (
              <div key={groupName}>
                <div className="flex items-center gap-2 text-[10px] sm:text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
                  <Clock3 className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
                  {groupName}
                </div>
                <div className="space-y-1">
                  {groupThreads.map((thread) => (
                    <div
                      key={thread.id}
                      className={`w-full flex items-center justify-between gap-2 sm:gap-3 rounded-xl px-2.5 sm:px-3 py-1.5 transition-colors ${
                        thread.id === activeThreadId
                          ? "bg-primary/10 text-primary"
                          : "text-foreground hover:bg-muted/60"
                      }`}
                    >
                      <button
                        onClick={() => {
                          setActiveThreadId(thread.id);
                          setInput("");
                          setShowSuggestions(true);
                        }}
                        className="flex min-w-0 flex-1 items-center gap-2 rounded-lg px-1.5 py-1.5 text-left"
                      >
                        <MessageSquareText className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" />
                        <span className="truncate text-sm sm:text-sm">{thread.title}</span>
                      </button>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <span className="hidden sm:inline text-xs text-muted-foreground">{thread.messages.length}</span>
                        <button
                          type="button"
                          onClick={() => setThreadToDelete(thread)}
                          aria-label={`Delete ${thread.title}`}
                          title="Delete chat"
                          className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:text-red-500 hover:bg-red-50 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))
          ) : (
            <div className="rounded-2xl border border-dashed border-border bg-background/70 px-4 py-6 text-center text-sm text-muted-foreground">
              No chats match "{searchQuery}".
            </div>
          )}
        </div>

        <div className="p-3 sm:p-4 border-t border-border">
          <button
            onClick={() => setShowClearConfirm(true)}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 py-2.5 sm:py-3 text-sm font-medium text-foreground hover:bg-muted transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Clear Current Chat
          </button>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="shrink-0 h-14 border-b border-border bg-card/80 backdrop-blur-sm flex items-center px-4 md:px-6 gap-3 z-10">
          <div className="flex items-center gap-2.5 flex-1 min-w-0">
            <button
              onClick={() => {
                if (window.innerWidth < 768) {
                  setMobileSidebarOpen((open) => !open);
                  return;
                }
                setSidebarOpen((open) => !open);
              }}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-background text-muted-foreground hover:text-foreground hover:bg-muted transition-colors shrink-0"
              aria-label="Toggle sidebar"
              title="Toggle sidebar"
            >
              <PanelLeft className="w-4 h-4" />
            </button>
            <div className="w-7 h-7 rounded-lg bg-primary flex items-center justify-center shadow-sm">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <div className="min-w-0">
              <span className="font-semibold text-sm text-foreground leading-none block">SupportAI</span>
              <span className="text-[11px] text-emerald-500 font-medium">● Online</span>
            </div>
            <div className="hidden sm:flex items-center gap-2 ml-4 px-3 py-1 rounded-full border border-border bg-background text-xs text-muted-foreground">
              <MessageSquareText className="w-3.5 h-3.5" />
              {activeThread?.title ?? "New Chat"}
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setDark((d) => !d)}
              title="Toggle theme"
              className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="max-w-5xl xl:max-w-6xl mx-auto px-3 sm:px-4 md:px-6 py-4 sm:py-5 lg:py-6">
            {!hasMessages ? (
              <EmptyState onSuggest={(q) => sendMessage(q)} />
            ) : (
              <div className="space-y-4 sm:space-y-5 lg:space-y-6">
                {messages.map((msg) => (
                  <MessageBubble
                    key={msg.id}
                    message={msg}
                    onFeedback={handleFeedback}
                    onRegenerate={handleRegenerate}
                  />
                ))}
                {loading && <TypingIndicator />}
                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </main>

        <div className="shrink-0 border-t border-border bg-card/80 backdrop-blur-sm">
          <div className="max-w-5xl xl:max-w-6xl mx-auto px-3 sm:px-4 md:px-6 py-3 sm:py-4">
            {showSuggestions && hasMessages && (
              <div className="flex flex-wrap items-center gap-2 mb-3">
                {SUGGESTED.slice(0, 4).map((q) => (
                  <button
                    key={q}
                    onClick={() => sendMessage(q)}
                    className="inline-flex items-center px-3 py-1.5 rounded-full border border-border bg-background text-xs text-foreground hover:bg-muted hover:border-primary/30 hover:text-primary transition-all duration-150 whitespace-nowrap"
                  >
                    {q}
                  </button>
                ))}
                <button
                  onClick={() => setShowSuggestions(false)}
                  className="ml-auto inline-flex items-center justify-center p-1 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            <div className="relative flex items-end gap-2 bg-input-background border border-border rounded-2xl px-3 sm:px-4 py-2.5 sm:py-3 shadow-sm focus-within:ring-2 focus-within:ring-primary/30 focus-within:border-primary/50 transition-all duration-150">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  autoResize();
                }}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about our products or services..."
                rows={1}
                disabled={loading}
                className="flex-1 bg-transparent resize-none outline-none text-sm text-foreground placeholder:text-muted-foreground leading-relaxed min-h-[24px] max-h-[160px] disabled:opacity-60 scrollbar-hide"
                style={{ fontFamily: "inherit" }}
              />
              <button
                onClick={() => sendMessage(input)}
                disabled={!input.trim() || loading}
                className="shrink-0 w-9 h-9 rounded-xl bg-primary text-white flex items-center justify-center hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150 shadow-sm"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <AlertDialog open={showClearConfirm} onOpenChange={setShowClearConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear current chat?</AlertDialogTitle>
            <AlertDialogDescription>
              All messages in this conversation will be removed. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                clearChat();
                setShowClearConfirm(false);
              }}
            >
              Clear
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={threadToDelete !== null}
        onOpenChange={(open) => {
          if (!open) {
            setThreadToDelete(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete chat?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove {threadToDelete?.title ?? "this chat"} from your history.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDeleteChat}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
