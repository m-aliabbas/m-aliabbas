"use client";
import { Card } from "components/Card";
import { LinkIcon } from "components/Icons";
import Image from "next/image";
import React from "react";

export default function CertificateCard({
  certificate,
}: {
  certificate: Certificate;
}): JSX.Element {
  const [isImageLoading, setImageLoading] = React.useState(true);

  return (
    <Card className="group flex flex-col items-center text-center p-6">
      {/* Certificate Image */}
      <div className="relative z-10 flex items-center justify-center">
        <Image
          src={`/images/${certificate.image.src}`}
          alt={certificate.image.alt}
          height={200}
          width={200}
          onLoad={() => setImageLoading(false)}
          className={`rounded-lg ${
            isImageLoading
              ? "blur-sm transition ease-in duration-100"
              : "blur-none transition ease-in duration-100"
          }`}
        />
      </div>

      {/* Certificate Title */}
      <h2 className="mt-4 text-lg font-semibold text-zinc-800 dark:text-zinc-100 group-hover:text-zinc-800 dark:group-hover:text-zinc-100">
        {certificate.title}
      </h2>

      {/* Certificate Authority */}
      <p className="mt-2 text-sm font-medium text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-500 dark:group-hover:text-zinc-400">
        {certificate.authority}
      </p>

      {/* Certificate Link */}
      <a
        href={certificate.link.href}
        target="_blank"
        rel="noopener noreferrer"
        className="relative z-10 mt-4 flex items-center text-sm font-medium text-teal-500 hover:text-teal-600 dark:text-teal-400 dark:hover:text-teal-300"
      >
        <LinkIcon className="h-5 w-5 flex-none" />
        <span className="ml-2">View Certificate</span>
      </a>
    </Card>
  );
}
