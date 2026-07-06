import { useEffect, useMemo, useState } from "react";
import { ChatInput } from "../components/ChatInput";
import { ChatWindow } from "../components/ChatWindow";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Navbar } from "../components/Navbar";
import { Sidebar } from "../components/Sidebar";
import { SuggestedQuestions } from "../components/SuggestedQuestions";
import { useChat } from "../hooks/useChat";
import { useIsMobile } from "../hooks/useIsMobile";

const THEME_STORAGE_KEY = "rag-frontend.theme";

function readPreferredTheme(): boolean {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === "dark") {
    return true;
  }
  if (stored === "light") {
    return false;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export default function Page() {
  const {
    sessions,
    activeSession,
    activeSessionId,
    isRequesting,
    createNewSession,
    clearActiveSession,
    deleteSession,
    setActiveSessionId,
    sendMessage,
    setFeedback,
    markTypingComplete,
  } = useChat();

  const isMobile = useIsMobile();
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [desktopSidebarOpen, setDesktopSidebarOpen] = useState(true);
  const [messageInput, setMessageInput] = useState("");
  const [sessionToDeleteId, setSessionToDeleteId] = useState<string | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [darkMode, setDarkMode] = useState<boolean>(() => readPreferredTheme());

  const hasMessages = (activeSession?.messages.length ?? 0) > 0;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem(THEME_STORAGE_KEY, darkMode ? "dark" : "light");
  }, [darkMode]);

  useEffect(() => {
    if (!isMobile) {
      setMobileSidebarOpen(false);
    }
  }, [isMobile]);

  useEffect(() => {
    document.body.style.overflow = isMobile && mobileSidebarOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [isMobile, mobileSidebarOpen]);

  const activeTitle = useMemo(() => activeSession?.title ?? "New Chat", [activeSession]);

  const sessionToDelete = useMemo(
    () => sessions.find((session) => session.id === sessionToDeleteId) ?? null,
    [sessions, sessionToDeleteId]
  );

  const onSendMessage = () => {
    if (!messageInput.trim()) {
      return;
    }

    sendMessage(messageInput);
    setMessageInput("");
  };

  const onSuggest = (question: string) => {
    sendMessage(question);
    setMessageInput("");
  };

  return (
    <div className="relative flex h-screen max-w-[100vw] flex-col overflow-hidden bg-background text-foreground supports-[height:100dvh]:h-dvh">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,rgba(37,99,235,0.12),transparent_45%),radial-gradient(circle_at_80%_0%,rgba(14,165,233,0.12),transparent_36%)]" />

      <Navbar
        title={activeTitle}
        darkMode={darkMode}
        onToggleTheme={() => setDarkMode((current) => !current)}
        onToggleSidebar={() => {
          if (isMobile) {
            setMobileSidebarOpen((open) => !open);
            return;
          }
          setDesktopSidebarOpen((open) => !open);
        }}
        onNewChat={() => {
          createNewSession();
          setMobileSidebarOpen(false);
        }}
      />

      <div className="flex min-h-0 w-full min-w-0 flex-1 overflow-hidden">
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          open={desktopSidebarOpen}
          mobileOpen={mobileSidebarOpen}
          onCloseMobile={() => setMobileSidebarOpen(false)}
          onSelectSession={setActiveSessionId}
          onRequestDeleteSession={setSessionToDeleteId}
          onRequestClearActive={() => setShowClearConfirm(true)}
        />

        <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-background">
          {hasMessages ? (
            <section className="hidden shrink-0 border-b border-border px-3 py-2 md:block lg:px-6">
              <div className="mx-auto w-full min-w-0 max-w-3xl">
                <SuggestedQuestions compact onSelect={onSuggest} />
              </div>
            </section>
          ) : null}

          <section className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain">
            <ChatWindow
              messages={activeSession?.messages ?? []}
              onFeedback={setFeedback}
              onTypingDone={markTypingComplete}
              onSuggest={onSuggest}
            />
          </section>

          <ChatInput
            value={messageInput}
            onChange={setMessageInput}
            isSending={isRequesting}
            onSend={onSendMessage}
          />
        </main>
      </div>

      <ConfirmDialog
        open={sessionToDeleteId !== null}
        title="Delete chat?"
        description={`This will permanently remove "${sessionToDelete?.title ?? "this chat"}" from your history.`}
        confirmLabel="Delete"
        cancelLabel="Cancel"
        onConfirm={() => {
          if (sessionToDeleteId) {
            deleteSession(sessionToDeleteId);
          }
          setSessionToDeleteId(null);
        }}
        onOpenChange={(open) => {
          if (!open) {
            setSessionToDeleteId(null);
          }
        }}
      />

      <ConfirmDialog
        open={showClearConfirm}
        title="Clear current chat?"
        description="All messages in this conversation will be removed. This action cannot be undone."
        confirmLabel="Clear"
        cancelLabel="Cancel"
        onConfirm={() => {
          clearActiveSession();
          setShowClearConfirm(false);
        }}
        onOpenChange={setShowClearConfirm}
      />
    </div>
  );
}
