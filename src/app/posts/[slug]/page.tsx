import { getPostBySlug, getAllPosts, markdownToHtml } from '@/lib/posts';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import Link from 'next/link';
import { notFound } from 'next/navigation';

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const posts = getAllPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export default async function PostPage({ params }: PageProps) {
  const { slug: _slug } = await params;
  const post = getPostBySlug(_slug);

  if (!post) {
    notFound();
  }

  const content = await markdownToHtml(post.content);

  return (
    <div className="container py-12 max-w-4xl" style={{ background: '#fff', borderRadius: '12px', marginTop: '2rem' }}>
      <article style={{ padding: '2rem' }}>
        <header className="mb-8">
          <Link 
            href="/" 
            className="text-sm mb-4 inline-block"
            style={{ color: '#666' }}
          >
            ← 返回首页
          </Link>
          <h1 className="text-4xl font-bold mb-4" style={{ color: '#333' }}>{post.title}</h1>
          <div className="flex items-center gap-4 text-sm" style={{ color: '#666' }}>
            <time dateTime={new Date(post.date).toISOString().split('T')[0]}>
              {format(new Date(post.date), 'yyyy年MM月dd日', { locale: zhCN })}
            </time>
            {post.tags.length > 0 && (
              <div className="flex gap-2">
                {post.tags.map((tag) => (
                  <Link
                    key={tag}
                    href={`/tags/${tag}`}
                    className="px-2 py-0.5 rounded text-xs transition-colors"
                    style={{ background: 'rgba(0, 0, 0, 0.05)', color: '#666' }}
                  >
                    {tag}
                  </Link>
                ))}
              </div>
            )}
          </div>
        </header>

        <div 
          className="prose prose-lg max-w-none"
          style={{ color: '#333' }}
          dangerouslySetInnerHTML={{ __html: content }}
        />
      </article>

      <footer className="mt-12 pt-8 border-t" style={{ borderColor: '#ddd' }}>
        <Link 
          href="/archives" 
          className="hover:underline"
          style={{ color: '#666' }}
        >
          ← 查看所有文章
        </Link>
      </footer>
    </div>
  );
}
