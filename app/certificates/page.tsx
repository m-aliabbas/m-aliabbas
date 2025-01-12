import SimpleLayout from "components/SimpleLayout";
import CertificatesPlaceholder from "components/skeleton/CertificatesPlaceholder"; // Placeholder for certificates
import { server } from "config";
import type { Metadata } from "next";
import { Suspense } from "react";
import ListCertificates from "./ListCertificates"; // Certificate listing component

export const metadata: Metadata = {
  title: "Certificates",
  description:
    "Over the years, I’ve earned several certificates showcasing my expertise in various fields. Here are the ones that I’m most proud of. Explore these to learn about my accomplishments.",
  openGraph: {
    title: "Certificates - Muhammad Ali Abbas",
    description:
      "Over the years, I’ve earned several certificates showcasing my expertise in various fields. Here are the ones that I’m most proud of. Explore these to learn about my accomplishments.",
    url: `${server}/certificates`,
    type: "website",
    site_name: "Muhammad Ali Abbas | Personal Website",
    images: [
      {
        url: `${server}/images/og-image.png`,
        alt: "Muhammad Ali Abbas",
        width: 1200,
        height: 630,
      },
    ],
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    site: "@mir_sazzat",
    creator: "@mir_sazzat",
    title: "Certificates - Muhammad Ali Abbas",
    description:
      "Over the years, I’ve earned several certificates showcasing my expertise in various fields. Here are the ones that I’m most proud of. Explore these to learn about my accomplishments.",
    images: [
      {
        url: `${server}/images/og-image.png`,
        alt: "Muhammad Ali Abbas",
        width: 1200,
        height: 630,
      },
    ],
  },
  alternates: {
    canonical: `${server}/certificates`,
    types: {
      "application/rss+xml": `${server}/feed.xml`,
    },
  },
};

export default function Certificates(): JSX.Element {
  return (
    <SimpleLayout
      title="Certifications that reflect my expertise and dedication."
      intro="Over the years, I’ve earned several certificates showcasing my expertise in various fields. Here are the ones that I’m most proud of. Explore these to learn about my accomplishments."
    >
      <div className="mt-16 sm:mt-20">
        <Suspense fallback={<CertificatesPlaceholder />}>
          {/* @ts-expect-error Server Component */}
          <ListCertificates />
        </Suspense>
      </div>
    </SimpleLayout>
  );
}
