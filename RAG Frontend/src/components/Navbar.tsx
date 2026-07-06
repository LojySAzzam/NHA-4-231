import { Menu, MessageSquareText, Plus } from "lucide-react";
import { ThemeSwitcher } from "./ThemeSwitcher";

interface NavbarProps {
  title: string;
  darkMode: boolean;
  onToggleTheme: () => void;
  onToggleSidebar: () => void;
  onNewChat: () => void;
}

export function Navbar({
  title,
  darkMode,
  onToggleTheme,
  onToggleSidebar,
  onNewChat,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-20 shrink-0 border-b border-border bg-background/95 backdrop-blur supports-[padding:max(0px)]:pt-[env(safe-area-inset-top)]">
      <div className="flex h-14 w-full min-w-0 items-center gap-2 px-3 sm:h-16 sm:gap-3 sm:px-4 lg:px-6">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition hover:bg-muted hover:text-foreground sm:h-10 sm:w-10"
          aria-label="Toggle chat history"
          title="Toggle chat history"
        >
          <Menu className="h-4 w-4" />
        </button>

        <div className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm sm:h-10 sm:w-10">
          <MessageSquareText className="h-4 w-4" />
        </div>

        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-foreground">RAG Assistant</p>
          <p className="truncate text-[11px] text-muted-foreground sm:text-xs">{title}</p>
        </div>

        <button
          type="button"
          onClick={onNewChat}
          className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-xl bg-primary px-2.5 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 sm:h-10 sm:gap-2 sm:px-3"
        >
          <Plus className="h-4 w-4" />
          <span className="hidden sm:inline">New Chat</span>
        </button>

        <ThemeSwitcher darkMode={darkMode} onToggle={onToggleTheme} />
      </div>
    </header>
  );
}
