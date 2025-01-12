// declare type for experience
declare type Experience = {
  title: string;
  company: string;
  companyURL: string;
  companyLogo: string;
  location: string;
  type: string;
  date: string;
  description: string;
  skills: string[];
};

declare type Education = {
  school: string;
  schoolURL: string;
  schoolLogo: string;
  schoolLocation: string;
  degree: string;
  major: string;
  minor: string;
  date: string;
  description: string;
  activitiesandsocieties: string[];
};

declare type Resource = {
  title: string;
  url: string;
  description: string;
  category: string;
};
declare type Certificate = {
  title: string; // The title of the certificate
  description: string; // A brief description of the certificate
  link: {
    href: string; // URL to verify or view the certificate
    label: string; // Text label for the link
  };
  image: {
    src: string; // Path to the certificate image
    alt: string; // Alternative text for the image
  };
};

declare type Project = {
  title: string;
  description: string;
  link: {
    href: string;
    label: string;
  };
  logo: {
    src: string;
    alt: string;
  };
};

declare type Course = {
  title: string;
  author: string;
  description: string;
  link: {
    href: string;
    label: string;
  };
  logo: {
    src: string;
    alt: string;
  };
  publishedDate: string;
  totalDuration: string;
};

declare type Color = {
  id: string;
  foreground: string;
  background: string;
};

declare type CalEvent = {
  title: string;
  description: string;
  location: string;
  date: string;
  startTime: string;
  endTime: string;
  color: Color;
  link: string;
};

declare type Day = {
  date: string;
  events: CalEvent[];
};
