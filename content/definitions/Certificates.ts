import { defineDocumentType } from "contentlayer/source-files";

export const Certificates = defineDocumentType(() => ({
  name: "certificate",
  filePathPattern: "certificates/*.mdx",
  contentType: "mdx",
  fields: {
    image: {
      type: "string",
      description: "The image of the certificate",
      required: true,
    },
    title: {
      type: "string",
      description: "The title of the certificate",
      required: true,
    },
    link: {
      type: "string",
      description: "The link to the certificate",
      required: true,
    },
    authority: {
      type: "string",
      description: "The certificate's issuing authority",
      required: true,
    },
    skills_learned: {
      type: "list",
      of: { type: "string" },
      description: "Skills learned from this certificate",
      required: true,
    },
  },
  computedFields: {
    slug: {
      type: "string",
      description: "The slug of the snippet",
      required: true,
      resolve: (doc) => doc._raw.sourceFileName.replace(/\.mdx$/, ""),
    },
  },
}));
