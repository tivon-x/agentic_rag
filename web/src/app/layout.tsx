import type { Metadata } from "next";
import Link from "next/link";

import { text } from "@/lib/i18n";

import "./globals.css";

export const metadata: Metadata = {
  title: "Agentic RAG 工作台",
  description: "面向生产前端的知识库构建与对话界面",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full">
        <div className="app-shell">
          <header className="border-b border-white/50 bg-white/70 backdrop-blur">
            <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
              <Link href="/" className="text-lg font-semibold tracking-tight text-slate-950">
                {text.nav.brand}
              </Link>
              <nav className="flex items-center gap-2">
                <Link href="/chat" className="nav-link">
                  {text.nav.chat}
                </Link>
                <Link href="/kb" className="nav-link">
                  {text.nav.kb}
                </Link>
              </nav>
            </div>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
