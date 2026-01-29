import { getAllSiteSettings } from '@/api/site-settings';
import { GrafanaMCPSettings, MCPSettings, ManagedMCPSettings } from '@/components/settings/IntegrationsSettings';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

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
        <Tabs defaultValue="database">
          <TabsList>
            <TabsTrigger value="database">Database</TabsTrigger>
            <TabsTrigger value="grafana">Grafana</TabsTrigger>
          </TabsList>
          <TabsContent value="database">
            <MCPSettings schema={settings} />
            <div className="mt-10" />
            <ManagedMCPSettings schema={settings} />
          </TabsContent>
          <TabsContent value="grafana">
            <GrafanaMCPSettings schema={settings} />
          </TabsContent>
        </Tabs>
      </div>
    </>
  );
}

export const dynamic = 'force-dynamic';

