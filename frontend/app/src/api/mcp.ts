import { requestUrl } from '@/lib/request';

export async function getMcpAgents (): Promise<string[]> {
  const res = await fetch(requestUrl('/api/v1/mcp/agents'), {
    credentials: 'include',
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return (data?.agents ?? []) as string[];
}

