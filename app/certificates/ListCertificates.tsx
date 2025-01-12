import CertificateCard from "components/CertificateCard";
import { server } from "config";
import fs, { promises as ps } from "fs";

// Get certificates from a local file or remote server
async function getListCertificates(): Promise<Certificate[]> {
  if (fs.existsSync("public/content/certificates.json")) {
    const res = await ps.readFile("public/content/certificates.json", "utf-8");
    const certificates: Certificate[] = JSON.parse(res);
    return certificates;
  }

  const certificates = fetch(`${server}/content/certificates.json`)
    .then((response) => response.json())
    .then((data) => {
      return data;
    });

  return certificates;
}

export default async function ListCertificates(): Promise<JSX.Element> {
  const certificates: Certificate[] = await getListCertificates();
  return (
    <ul
      role="list"
      className="grid grid-cols-1 gap-x-12 gap-y-16 sm:grid-cols-2 lg:grid-cols-3"
    >
      {certificates.map((certificate) => (
        <CertificateCard
          as="li"
          certificate={certificate}
          key={certificate.title}
        />
      ))}
    </ul>
  );
}
