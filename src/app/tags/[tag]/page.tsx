import Link from 'next/link';
import { getAllTags, getPostsByTag } from '@/lib/posts';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import { notFound } from 'next/navigation';

interface PageProps {
  params: Promise<{ tag: string }>;
}

export async function generateStaticParams() {
  const tags = getAllTags();
  return tags.map((tag) => ({ tag }));
}

export default async function TagPage({ params }: PageProps) {
  const { tag } = await params;
  const posts = getPostsByTag(tag);

  if (posts.length === 0) {
    notFound();
  }

  return (
    <div className="container py-12">
      <header className="mb-8">
        <Link
          href="/tags"
          className="text-sm text-secondary hover:text-primary mb-4 inline-block"
        >
          ← 返回标签列表
        </Link>
        <h1 className="text-4xl font-bold mb-4">
          标签: <span className="text-primary">{tag}</span>
        </h1>
        <p className="text-secondary">包含此标签的文章 ({posts.length} 篇)</p>
      </header>

      <div className="space-y-4">
        {posts.map((post) => (
          <article key={post.slug} className="card hover:border-primary">
            <Link href={`/posts/${post.slug}`}>
              <h3 className="text-xl font-semibold mb-2 hover:text-primary">
                {post.title}
              </h3>
              <time className="text-secondary text-sm">
                {format(new Date(post.date), 'yyyy年MM月dd日', { locale: zhCN })}
              </time>
            </Link>
          </article>
        ))}
      </div>
    </div>
  );
}
