import { ImageIcon } from "components/Icons";

export default function CertificatesPlaceholder() {
  return (
    <ul
      role="list"
      className="grid grid-cols-1 gap-x-12 gap-y-16 sm:grid-cols-2 lg:grid-cols-3"
    >
      {[...Array(6)].map((_, i) => (
        <li
          key={i}
          className="group relative flex flex-col items-center animate-pulse w-full"
        >
          {/* Placeholder for Certificate Image */}
          <div className="h-48 w-48 bg-gray-300 dark:bg-gray-700 rounded-lg mb-4 flex items-center justify-center">
            <CertificateIcon className="h-12 w-12 text-gray-200 dark:text-gray-600" />
          </div>

          {/* Placeholder for Certificate Title */}
          <div className="h-4 w-3/4 bg-gray-300 dark:bg-gray-600 rounded mb-4"></div>

          {/* Placeholder for Certificate Link */}
          <div className="h-3 w-1/2 bg-gray-300 dark:bg-gray-600 rounded"></div>
        </li>
      ))}
    </ul>
  );
}
