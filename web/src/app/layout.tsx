import type { Metadata } from "next";
import Link from "next/link";

import { text } from "@/lib/i18n";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Paper Index",
    template: "%s | Paper Index",
  },
  description: "可搜索、可校正、可回到 PDF 原页的个人论文库。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="zh-CN"
      className="h-full antialiased"
      data-scroll-behavior="smooth"
    >
      <body className="min-h-full">
        <a href="#main-content" className="skip-link">
          跳到正文
        </a>
        <div className="app-shell">
          <header className="masthead">
            <div className="masthead-inner">
              <Link href="/" className="brand-lockup">
                <span className="brand-name">{text.nav.brand}</span>
                <span className="brand-caption">LOCAL PAPER LIBRARY</span>
              </Link>
              <nav aria-label="主导航" className="masthead-nav">
                <Link href="/library" className="nav-link">
                  {text.nav.library}
                </Link>
                <Link href="/search" className="nav-link">
                  {text.nav.search}
                </Link>
                <Link href="/chat" className="nav-link">
                  {text.nav.chat}
                </Link>
              </nav>
            </div>
          </header>
          {children}
          <footer className="site-footer">
            <span>证据优先，回到原页。</span>
            <span className="font-mono text-[0.68rem]">FIXED RAG / LOCAL FIRST</span>
          </footer>
        </div>
      </body>
    </html>
  );
}
