import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="border-t border-[var(--color-border-light)] bg-[var(--color-bg)]">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="flex flex-col md:flex-row justify-between items-start gap-8">
          <div>
            <p className="text-lg font-semibold tracking-tight mb-1">FaceMaxx</p>
            <p className="text-sm text-[var(--color-text-tertiary)]">Personal color intelligence.</p>
          </div>

          <div className="flex gap-8 text-sm text-[var(--color-text-secondary)]">
            <a href="/#how-it-works">How it works</a>
            <Link to="/about">About</Link>
            <Link to="/privacy">Privacy</Link>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-[var(--color-border-light)]">
          <p className="text-xs text-[var(--color-text-tertiary)]">
            &copy; {new Date().getFullYear()} FaceMaxx. Your photo is processed locally and not stored.
          </p>
        </div>
      </div>
    </footer>
  );
}
