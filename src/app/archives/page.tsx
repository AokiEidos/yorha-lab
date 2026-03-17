import Link from 'next/link';
import { getAllPosts } from '@/lib/posts';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

export default function ArchivesPage() {
  const posts = getAllPosts();

  // 按年份分组
  const postsByYear = posts.reduce((acc, post) => {
    const year = new Date(post.date).getFullYear();
    if (!acc[year]) {
      acc[year] = [];
    }
    acc[year].push(post);
    return acc;
  }, {} as Record<number, typeof posts>);

  const years = Object.keys(postsByYear).sort((a, b) => Number(b) - Number(a));

  return (
    <div className="container py-12">
      <h1 className="text-4xl font-bold mb-8">归档</h1>

      {posts.length === 0 ? (
        <div className="card">
          <p className="text-secondary">暂无文章...</p>
        </div>
      ) : (
        <div className="space-y-8">
          {years.map((year) => (
            <section key={year}>
              <h2 className="text-2xl font-semibold mb-4">{year}</h2>
              <ul className="space-y-3">
                {postsByYear[Number(year)].map((post) => (
                  <li key={post.slug}>
                    <Link
                      href={`/posts/${post.slug}`}
                      className="block p-4 card hover:border-primary"
                    >
                      <time className="text-secondary text-sm">
                        {format(new Date(post.date), 'MM-dd', { locale: zhCN })}
                      </time>
                      <span className="ml-4">{post.title}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
