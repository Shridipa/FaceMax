import { Link } from "react-router-dom";

export default function Privacy() {
  return (
    <main className="pt-16 min-h-screen">
      <div className="max-w-2xl mx-auto px-6 py-16 md:py-24">
        <h1 className="text-3xl md:text-4xl font-serif mb-8">Privacy</h1>

        <div className="space-y-6 text-sm text-[var(--color-text-secondary)] leading-relaxed">
          <div className="p-4 bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-xl">
            <p className="font-medium text-[var(--color-text-primary)]">
              Your photo is processed in real time and is not stored on our servers.
            </p>
          </div>

          <h2 className="text-lg font-semibold text-[var(--color-text-primary)] pt-2">How your photo is handled</h2>

          <ul className="space-y-2">
            <li>When you upload a photo, it is sent to our server for analysis only.</li>
            <li>The photo is processed by our computer vision pipeline to extract color information.</li>
            <li>The photo is deleted immediately after processing.</li>
            <li>We do not store, cache, or share your photo.</li>
            <li>We do not use your photo for training or any other purpose.</li>
          </ul>

          <h2 className="text-lg font-semibold text-[var(--color-text-primary)] pt-2">What we store</h2>

          <p>
            We do not store any personal data. Analysis results are kept only in your
            browser&apos;s session storage and are automatically cleared when you close the tab.
          </p>

          <h2 className="text-lg font-semibold text-[var(--color-text-primary)] pt-2">Third parties</h2>

          <p>
            We do not share any data with third parties. Your photo never leaves our
            processing pipeline.
          </p>
        </div>

        <div className="mt-12">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-6 py-3 border border-[var(--color-border)] text-sm font-medium rounded-full hover:bg-[var(--color-surface-muted)] transition-colors"
          >
            Back to home
          </Link>
        </div>
      </div>
    </main>
  );
}
