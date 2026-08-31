"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { text } from "@/lib/i18n";

const links = [
  { href: "/library", label: text.nav.library },
  { href: "/search", label: text.nav.search },
  { href: "/chat", label: text.nav.chat },
  { href: "/evaluation", label: "评测" },
] as const;

function isCurrentPath(pathname: string | null, href: string) {
  return pathname === href || (href !== "/" && pathname?.startsWith(`${href}/`));
}

export default function EditorialNav() {
  const pathname = usePathname();

  return (
    <nav aria-label="主导航" className="masthead-nav">
      {links.map(({ href, label }) => (
        <Link
          key={href}
          href={href}
          className="nav-link"
          aria-current={isCurrentPath(pathname, href) ? "page" : undefined}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
