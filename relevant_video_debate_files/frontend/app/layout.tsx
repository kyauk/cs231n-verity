import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Triage Refinery Workspace",
  description: "Failure ingestion and capsule workspace for triage refinement."
};

type RootLayoutProps = {
  children: React.ReactNode;
};

export default function RootLayout({ children }: RootLayoutProps): JSX.Element {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
