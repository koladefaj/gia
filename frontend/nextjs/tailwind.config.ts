import type { Config } from 'tailwindcss';

/**
 * Tokens live as CSS variables (see globals.css) so a single `data-theme`
 * attribute on <html> flips the whole surface between light and dark. Tailwind
 * utilities read those variables here, and the `dark:` variant is wired to the
 * same attribute for the rare spot where a utility is cleaner than a token.
 */
const config: Config = {
  darkMode: ['selector', '[data-theme="dark"]'],
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        field: 'rgb(var(--field) / <alpha-value>)',
        ink: 'rgb(var(--ink) / <alpha-value>)',
        'ink-soft': 'rgb(var(--ink-soft) / <alpha-value>)',
        'ink-faint': 'rgb(var(--ink-faint) / <alpha-value>)',
        accent: 'rgb(var(--accent) / <alpha-value>)',
        hairline: 'rgb(var(--hairline) / <alpha-value>)',
      },
      fontFamily: {
        display: ['var(--font-display)', 'system-ui', 'sans-serif'],
        sans: ['var(--font-body)', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        panel: '20px',
      },
      maxWidth: {
        prose: '65ch',
      },
    },
  },
  plugins: [],
};

export default config;
