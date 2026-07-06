import { MessageSquareText, Search, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { ChatSession } from "../types/chat";

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string;
  open: boolean;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  onSelectSession: (id: string) => void;
  onRequestDeleteSession: (id: string) => void;
  onRequestClearActive: () => void;
}

export function Sidebar({
  sessions,
  activeSessionId,
  open,
  mobileOpen,
  onCloseMobile,
  onSelectSession,
  onRequestDeleteSession,
  onRequestClearActive,
}: SidebarProps) {
  const [query, setQuery] = useState("");

  const visibleSessions = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return sessions;
    }

    return sessions.filter((session) => {
      if (session.title.toLowerCase().includes(normalized)) {
        return true;
      }

      return session.messages.some((message) =>
        message.content.toLowerCase().includes(normalized)
      );
    });
  }, [sessions, query]);

  return (
    <>
      {mobileOpen ? (
        <button
          type="button"
          className="fixed inset-x-0 bottom-0 top-14 z-30 bg-black/40 md:hidden"
          onClick={onCloseMobile}
          aria-label="Close sidebar"
        />
      ) : null}

      <aside
        className={`fixed left-0 top-14 z-40 flex h-[calc(100dvh-3.5rem)] w-[min(100%,20rem)] max-w-[88vw] flex-col border-r border-border bg-card shadow-xl transition-transform duration-300 md:static md:z-0 md:h-full md:w-72 md:max-w-none md:shrink-0 md:self-stretch md:shadow-none lg:w-80 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        } ${open ? "md:flex" : "md:hidden"}`}
      >
        <div className="shrink-0 border-b border-border p-3 sm:p-4">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search chat history"
              className="w-full min-w-0 rounded-xl border border-border bg-background py-2 pl-9 pr-3 text-sm text-foreground outline-none ring-primary/20 placeholder:text-muted-foreground focus:ring-2"
            />
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-3 sm:p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Chat History
          </p>

          <div className="space-y-1.5">
            {visibleSessions.map((session) => (
              <article
                key={session.id}
                className={`group flex min-w-0 items-center gap-2 rounded-xl border px-2.5 py-2 transition ${
                  activeSessionId === session.id
                    ? "border-primary/30 bg-accent/60"
                    : "border-transparent hover:border-border hover:bg-muted/60"
                }`}
              >
                <button
                  type="button"
                  onClick={() => {
                    onSelectSession(session.id);
                    onCloseMobile();
                  }}
                  className="min-w-0 flex-1 text-left"
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <MessageSquareText className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <p className="truncate text-sm font-medium text-foreground">{session.title}</p>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {session.messages.length} messages
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => onRequestDeleteSession(session.id)}
                  className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition hover:bg-red-100 hover:text-red-700 dark:hover:bg-red-950/40 dark:hover:text-red-300"
                  aria-label={`Delete ${session.title}`}
                  title="Delete chat"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </article>
            ))}
          </div>
        </div>

        <div className="shrink-0 border-t border-border p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:p-4">
          <button
            type="button"
            onClick={onRequestClearActive}
            className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition hover:bg-muted"
          >
            Clear Current Chat
          </button>
        </div>
      </aside>
    </>
  );
}
