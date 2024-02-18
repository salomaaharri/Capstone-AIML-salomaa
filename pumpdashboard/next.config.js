// next.config.js
const nextConfig = {
    webpack: (config, { isServer }) => {
      config.module.rules.push({
        test: /\.svg$/,
        use: ['@svgr/webpack'] // Only use @svgr/webpack
      });
      return config;
    }
}
  
module.exports = nextConfig;
  