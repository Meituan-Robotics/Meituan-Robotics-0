import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const imageUrl = `${protocol}://${host}/og.png`;

  return {
    title: "Meituan-Robotics-0",
    description:
      "A Vision-Language-Action foundation model for precise bimanual desktop manipulation.",
    robots: {
      index: false,
      follow: false,
      nocache: true,
      googleBot: {
        index: false,
        follow: false,
        noimageindex: true,
      },
    },
    icons: {
      icon: "/favicon.png",
      shortcut: "/favicon.png",
    },
    openGraph: {
      title: "Meituan-Robotics-0",
      description: "Vision-Language-Action for Desktop Manipulation",
      type: "website",
      images: [{ url: imageUrl, width: 1672, height: 941, alt: "Meituan-Robotics-0" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Meituan-Robotics-0",
      description: "Vision-Language-Action for Desktop Manipulation",
      images: [imageUrl],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
