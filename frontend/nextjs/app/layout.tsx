import type { Metadata, Viewport } from 'next';
import { Bricolage_Grotesque, Manrope } from 'next/font/google';
import './globals.css';

// Display: Bricolage Grotesque — a contemporary grotesque with real character
// for headlines. Body / UI: Manrope — quiet, legible, carries small text well.
const display = Bricolage_Grotesque({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-display',
  display: 'swap',
});

const body = Manrope({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600'],
  variable: '--font-body',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Gia - Talk to your music',
  description:
    'A voice companion that listens, learns your taste, and plays what fits the moment. Hands-free, powered by your Spotify.',
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f6f3ee' },
    { media: '(prefers-color-scheme: dark)', color: '#090909' },
  ],
};

// Set the theme attribute before first paint so there's no light/dark flash.
const themeInit = `(function(){try{var k=localStorage.getItem('gia_theme');var d=window.matchMedia('(prefers-color-scheme: dark)').matches;document.documentElement.setAttribute('data-theme',(k==='light'||k==='dark')?k:(d?'dark':'light'));}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
