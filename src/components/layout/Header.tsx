import { Link } from "react-router-dom";

export default function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[var(--color-bg)]/90 backdrop-blur-md border-b border-[var(--color-border-light)]">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-lg font-semibold tracking-tight text-[var(--color-text-primary)]">
            FaceMaxx
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-8">
          <a href="/#how-it-works" className="text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors">
            How it works
          </a>
          <Link
            to="/outfit-studio"
            className="text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
          >
            Outfit Studio
          </Link>
          <Link
            to="/analyze"
            className="text-sm font-medium px-5 py-2 bg-white text-[var(--color-text-primary)] rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
          >
            Analyze my colors
          </Link>
        </nav>

        <Link
          to="/analyze"
          className="md:hidden text-sm font-medium px-4 py-2 bg-white text-[var(--color-text-primary)] rounded-full border border-[var(--color-border)]"
        >
          Analyze
        </Link>
      </div>
    </header>
  );
}
