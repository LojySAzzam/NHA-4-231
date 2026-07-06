import { Moon, Sun } from "lucide-react";

interface ThemeSwitcherProps {
  darkMode: boolean;
  onToggle: () => void;
}

export function ThemeSwitcher({ darkMode, onToggle }: ThemeSwitcherProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition hover:text-foreground hover:bg-muted"
      aria-label="Toggle dark mode"
      title="Toggle dark mode"
    >
      {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}
