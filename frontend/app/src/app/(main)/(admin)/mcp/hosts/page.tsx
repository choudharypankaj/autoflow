import { getAllSiteSettings } from '@/api/site-settings';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { McpHostsTabs } from '@/components/settings/McpHostsTabs';

export default async function AdminMcpHostsPage () {
  const settings = await getAllSiteSettings();
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'MCP Hosts' },
        ]}
      />
      <div className="max-w-screen-md">
        <McpHostsTabs settings={settings} />
      </div>
    </>
  );
}

export const dynamic = 'force-dynamic';

