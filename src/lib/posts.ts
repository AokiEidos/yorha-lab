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
  content = content.replace(/\$([^$\n]+)\$/g, (match, formula) => {
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

  // 读取 content/posts 下的所有 .md 文件（平铺结构）
  const files = fs.readdirSync(postsDirectory)
    .filter(f => f.endsWith('.md'));

  const allPostsData = files
    .map((fileName) => {
      const fullPath = path.join(postsDirectory, fileName);
      const fileContents = fs.readFileSync(fullPath, 'utf8');
      const { data, content } = matter(fileContents);
      const { title: extractedTitle, excerpt } = extractTitleAndExcerpt(content);
      
      // slug 从文件名去掉 .md
      const slug = fileName.replace(/\.md$/, '');

      return {
        slug,
        title: data.title || extractedTitle || '未命名论文',
        date: data.date || new Date().toISOString().split('T')[0],
        tags: data.tags || [],
        excerpt: excerpt || content.substring(0, 200).replace(/[#*`\n]/g, ' ').trim(),
        readingTime: estimateReadingTime(content),
        views: 0,
        isPinned: data.isPinned || false,
      };
    })
    .filter(Boolean) as PostMeta[];

  return allPostsData.sort((a, b) => {
    if (a.isPinned && !b.isPinned) return -1;
    if (!a.isPinned && b.isPinned) return 1;
    return a.date < b.date ? 1 : -1;
  });
}

export function getPostBySlug(slug: string): Post | null {
  // 优先查找 flat .md 文件
  const mdPath = path.join(postsDirectory, `${slug}.md`);
  const fullPath = fs.existsSync(mdPath) ? mdPath : path.join(postsDirectory, slug, 'README.md');
  
  if (!fs.existsSync(fullPath)) {
    return null;
  }

  const fileContents = fs.readFileSync(fullPath, 'utf8');
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

export async function markdownToHtml(markdown: string): Promise<string> {
  // 先渲染 LaTeX 公式
  markdown = renderLatex(markdown);
  
  // 转换为 HTML（支持 GFM 表格等）
  const result = await remark()
    .use(remarkGfm)  // 支持表格
    .use(html, { sanitize: false })
    .process(markdown);
  
  return result.toString();
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
