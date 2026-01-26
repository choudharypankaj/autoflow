import { getAllSiteSettings } from '@/api/site-settings';
import { MCPSettings, ManagedMCPSettings } from '@/components/settings/IntegrationsSettings';
import { AdminPageHeading } from '@/components/admin-page-heading';

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
        <MCPSettings schema={settings} />
        <div className="mt-10" />
        <ManagedMCPSettings schema={settings} />
      </div>
    </>
  );
}

export const dynamic = 'force-dynamic';

