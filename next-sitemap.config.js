/** @type {import('next-sitemap').IConfig} */
module.exports = {
  siteUrl: process.env.SITE_URL || "https://m-aliabbas.vercel.app/",
  generateRobotsTxt: true,
  exclude: ['/server-sitemap.xml'],
  robotsTxtOptions: {
    policies: [
      {
        userAgent: "*",
        allow: "/",
      },
    ],
    additionalSitemaps: [
      `${process.env.SITE_URL || "https://m-aliabbas.vercel.app/"}/sitemap.xml`,
      `${process.env.SITE_URL || "https://m-aliabbas.vercel.app/"}/server-sitemap.xml`,
    ],
  },
};
