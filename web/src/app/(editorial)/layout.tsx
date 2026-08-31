import Link from "next/link";

import EditorialNav from "@/components/EditorialNav";
import { text } from "@/lib/i18n";

export default function EditorialLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="masthead-inner">
          <Link href="/" className="brand-lockup">
            <span className="brand-name">{text.nav.brand}</span>
            <span className="brand-caption">LOCAL PAPER LIBRARY</span>
          </Link>
          <EditorialNav />
        </div>
      </header>
      {children}
      <footer className="site-footer">
        <span>证据优先，回到原页。</span>
        <span className="font-mono text-[0.68rem]">FIXED RAG / LOCAL FIRST</span>
      </footer>
    </div>
  );
}
