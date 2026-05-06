'use client';

import { useEffect } from 'react';

export function MermaidRenderer() {
  useEffect(() => {
    async function renderMermaid() {
      const mermaidBlocks = document.querySelectorAll('.mermaid');
      if (mermaidBlocks.length === 0) return;

      const mermaid = (await import('mermaid')).default;
      mermaid.initialize({
        startOnLoad: false,
        theme: 'default',
        securityLevel: 'loose',
      });

      for (const block of mermaidBlocks) {
        const id = `mermaid-${Math.random().toString(36).slice(2, 10)}`;
        const code = block.textContent || '';
        try {
          const { svg } = await mermaid.render(id, code);
          block.innerHTML = svg;
        } catch (e) {
          console.error('Mermaid render failed:', e);
        }
      }
    }
    renderMermaid();
  }, []);

  return null;
}