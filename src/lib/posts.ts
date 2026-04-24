import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';
import remarkGfm from 'remark-gfm';
import katex from 'katex';

const postsDirectory = path.join(process.cwd(), 'content/posts');

export interface PostMeta {
  slug: string;
  title: string;
  date: string;
  tags: string[];
  excerpt: string;
  cover?: string;
  views?: number;
  readingTime?: number;
  isPinned?: boolean;
  hidden?: boolean;
}

export interface Post extends PostMeta {
  content: string;
}

// 从论文 README.md 中提取标题和摘要
function extractTitleAndExcerpt(content: string): { title: string; excerpt: string } {
  // 1. 优先从 frontmatter 提取标题
  const titleMatch = content.match(/^title:\s*["']?(.*?)["']?\s*$/m);
  let title = titleMatch ? titleMatch[1].trim() : '';
  
  // 2. 如果 frontmatter 没有，尝试从 # 📄 标题行提取
  if (!title) {
    const headingMatch = content.match(/^#\s*📄\s*(.+)$/m);
    if (headingMatch) {
      title = headingMatch[1].replace(/ - 深度解读报告$/, '').trim();
    }
  }
  
  // 3. 摘要优先从"核心主旨"章节提取
  let excerpt = '';
  const coreGistMatch = content.match(/##?\s*0\.\s*核心主旨.*?[\n\r]+([^\n#]+)/);
  if (coreGistMatch) {
    excerpt = coreGistMatch[1].trim().substring(0, 200);
  }
  
  return { title: title || '未命名论文', excerpt };
}

function estimateReadingTime(content: string): number {
  const words = content.length;
  return Math.max(1, Math.ceil(words / 1000));
}

// 渲染 KaTeX 公式
function renderLatex(content: string): string {
  // 渲染块级公式 $$...$$
  content = content.replace(/\$\$([^$]+)\$\$/g, (match, formula) => {
    try {
      return katex.renderToString(formula.trim(), {
        throwOnError: false,
        displayMode: true,
      });
    } catch (e) {
      return match;
    }
  });
  
  // 渲染行内公式 $...$
  // 非贪婪（+?)，防止跨越多个 $...$ 匹配
  content = content.replace(/\$([^$\n]+?)\$/g, (match, formula) => {
    try {
      return katex.renderToString(formula.trim(), {
        throwOnError: false,
        displayMode: false,
      });
    } catch (e) {
      return match;
    }
  });
  
  return content;
}

export function getAllPosts(): PostMeta[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  // 递归读取 content/posts 下的所有 .md 文件（含子目录）
  const files: string[] = [];
  function walkDir(dir: string) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        walkDir(path.join(dir, entry.name));
      } else if (entry.name.endsWith('.md')) {
        files.push(path.join(dir, entry.name));
      }
    }
  }
  walkDir(postsDirectory);

  const allPostsData = files
    .map((fullPath) => {
      const fileContents = fs.readFileSync(fullPath, 'utf8');
      const { data, content } = matter(fileContents);
      const { title: extractedTitle, excerpt } = extractTitleAndExcerpt(content);

      // slug: 相对于 postsDirectory 的路径，去掉 .md
      const slug = path.relative(postsDirectory, fullPath).replace(/\.md$/, '');

      return {
        slug,
        title: data.title || extractedTitle || '未命名论文',
        date: data.date || new Date().toISOString().split('T')[0],
        tags: data.tags || [],
        excerpt: excerpt || content.substring(0, 200).replace(/[#*`\n]/g, ' ').trim(),
        readingTime: estimateReadingTime(content),
        views: 0,
        isPinned: data.isPinned || false,
        hidden: data.hidden || false,
      };
    })
    .filter(Boolean) as PostMeta[];

  return allPostsData
    .filter(p => !p.hidden)
    .sort((a, b) => {
    if (a.isPinned && !b.isPinned) return -1;
    if (!a.isPinned && b.isPinned) return 1;
    return a.date < b.date ? 1 : -1;
  });
}

export function getAllPostsIncludingHidden(): PostMeta[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const files: string[] = [];
  function walkDir(dir: string) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        walkDir(path.join(dir, entry.name));
      } else if (entry.name.endsWith('.md')) {
        files.push(path.join(dir, entry.name));
      }
    }
  }
  walkDir(postsDirectory);

  return files
    .map((fullPath) => {
      const fileContents = fs.readFileSync(fullPath, 'utf8');
      const { data, content } = matter(fileContents);
      const { title: extractedTitle, excerpt } = extractTitleAndExcerpt(content);
      const slug = path.relative(postsDirectory, fullPath).replace(/\.md$/, '');

      return {
        slug,
        title: data.title || extractedTitle || '未命名论文',
        date: data.date || new Date().toISOString().split('T')[0],
        tags: data.tags || [],
        excerpt: excerpt || content.substring(0, 200).replace(/[#*`\n]/g, ' ').trim(),
        readingTime: estimateReadingTime(content),
        views: 0,
        isPinned: data.isPinned || false,
        hidden: data.hidden || false,
      };
    })
    .filter(Boolean) as PostMeta[];
}

export function getPostBySlug(slug: string): Post | null {
  // 子目录 README 优先（支持 /posts/diffusion → diffusion/README.md）
  const readmePath = path.join(postsDirectory, slug, 'README.md');
  if (fs.existsSync(readmePath)) {
    const fileContents = fs.readFileSync(readmePath, 'utf8');
    const { data, content } = matter(fileContents);
    const { title, excerpt } = extractTitleAndExcerpt(content);
    return {
      slug,
      title: title || data.title || '未命名',
      date: data.date || new Date().toISOString().split('T')[0],
      tags: data.tags || [],
      excerpt,
      content,
      readingTime: estimateReadingTime(content),
      views: 0,
    };
  }

  // 平铺 .md 文件
  const directPath = path.join(postsDirectory, `${slug}.md`);
  if (fs.existsSync(directPath)) {
    const fileContents = fs.readFileSync(directPath, 'utf8');
    const { data, content } = matter(fileContents);
    const { title, excerpt } = extractTitleAndExcerpt(content);
    const arxivMatch = slug.match(/arxiv-(\d+\.\d+)/);
    const arxivId = arxivMatch ? arxivMatch[1] : '';
    return {
      slug,
      title: title || data.title || '未命名论文',
      date: data.date || new Date().toISOString().split('T')[0],
      tags: data.tags || [arxivId ? `ArXiv ${arxivId}` : '论文'].filter(Boolean),
      excerpt,
      content,
      readingTime: estimateReadingTime(content),
      views: 0,
    };
  }

  return null;
}

export async function markdownToHtml(markdown: string): Promise<string> {
  // 先用占位符保护 LaTeX 公式，防止 remark 把 _ 转成 <em>
  const mathPlaceholders: string[] = [];
  
  // 保护块级公式 $$...$$
  markdown = markdown.replace(/\$\$([^$]+)\$\$/g, (match) => {
    const idx = mathPlaceholders.length;
    mathPlaceholders.push(match);
    return `MATHPLACEHOLDER${idx}ENDMATH`;
  });
  
  // 保护行内公式 $...$
  markdown = markdown.replace(/\$([^$\n]+?)\$/g, (match) => {
    const idx = mathPlaceholders.length;
    mathPlaceholders.push(match);
    return `MATHPLACEHOLDER${idx}ENDMATH`;
  });
  
  // 转换为 HTML（支持 GFM 表格等）
  const result = await remark()
    .use(remarkGfm)  // 支持表格
    .use(html, { sanitize: false })
    .process(markdown);
  
  let htmlContent = result.toString();
  
  // 还原占位符并渲染 LaTeX
  htmlContent = htmlContent.replace(/MATHPLACEHOLDER(\d+)ENDMATH/g, (_, idx) => {
    const original = mathPlaceholders[parseInt(idx)];
    return renderLatex(original);
  });
  
  return htmlContent;
}

export function getAllTags(): string[] {
  const posts = getAllPosts();
  const tagSet = new Set<string>();
  
  posts.forEach((post) => {
    post.tags.forEach((tag) => tagSet.add(tag));
  });
  
  return Array.from(tagSet).sort();
}

export function getPostsByTag(tag: string): PostMeta[] {
  const posts = getAllPosts();
  return posts.filter((post) => post.tags.includes(tag));
}
