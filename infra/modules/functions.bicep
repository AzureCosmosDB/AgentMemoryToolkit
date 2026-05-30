// Function app on Flex Consumption (Python 3.11) + storage + App Insights
// + Log Analytics workspace. Identity-based connections; no secrets in app
// settings.

@description('Function app name.')
param functionAppName string

@description('Storage account name (lowercase, <=24).')
param storageAccountName string

@description('App Insights component name.')
param appInsightsName string

@description('Log Analytics workspace name.')
param logAnalyticsName string

@description('App Service plan name (Flex Consumption).')
param planName string

@description('Azure region.')
param location string

@description('User-assigned managed identity resource id.')
param uamiResourceId string

@description('User-assigned managed identity client id.')
param uamiClientId string

@description('Cosmos account endpoint for identity-based binding.')
param cosmosEndpoint string

@description('Cosmos database name.')
param cosmosDatabase string = 'ai_memory'

@description('Memories container name.')
param cosmosContainer string = 'memories'

@description('Turns container name.')
param cosmosTurnsContainer string = 'memories_turns'

@description('Summaries container name.')
param cosmosSummariesContainer string = 'memories_summaries'

@description('Lease container name.')
param cosmosLeaseContainer string = 'leases'

@description('Counters container name.')
param cosmosCountersContainer string = 'counter'

@description('AI Foundry endpoint URL.')
param aiFoundryEndpoint string

@description('Embedding model deployment name.')
param embeddingDeploymentName string = 'text-embedding-3-large'

@description('Embedding output dimensions. MUST match the dimensions configured in the Cosmos memories container vectorEmbeddingPolicy (default 1536).')
param embeddingDimensions int = 1536

@description('LLM model deployment name.')
param chatDeploymentName string = 'gpt-4o-mini'

@description('Azure OpenAI REST API version pinned for both chat and embedding clients. Always supplied by main.bicep — declared here without a default so the wiring stays explicit.')
param azureOpenAiApiVersion string

// --- Function-app threshold / batching knobs ------------------------------
//
// All knobs are surfaced as Bicep params in main.bicep (bound to
// `${THREAD_SUMMARY_EVERY_N=10}` etc. in main.parameters.json) so customers
// can override them via `azd env set ...` before `azd up`. The defaults live
// in main.bicep — these module params are declared without defaults so
// main.bicep stays the single source of truth.

@description('Run thread-summary orchestration every N turns within a (user_id, thread_id). 0 = disabled.')
param threadSummaryEveryN int

@description('Run fact / episodic / procedural extraction every N turns within a (user_id, thread_id). 0 = disabled.')
param factExtractionEveryN int

@description('Run dedup once per N fact-extraction batches. Effective cadence = factExtractionEveryN * dedupEveryN turns.')
param dedupEveryN int

@description('Run user-summary orchestration every N turns from a given user_id across all threads. 0 = disabled.')
param userSummaryEveryN int

@description('Maximum number of change-feed items processed per orchestration batch.')
param maxBatchSize int

@description('Backend that owns processing. `durable` = the function-app fleet owns processing; SDK clients pointed at the same container will skip auto-triggering. `inprocess` = SDK owns processing, function-app skips.')
@allowed([
  'durable'
  'inprocess'
])
param memoryProcessorOwner string

@description('Tags to apply.')
param tags object = {}

// --- Log Analytics + App Insights -----------------------------------------

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// --- Storage account ------------------------------------------------------

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    // Identity-only design: Durable Functions, AzureWebJobsStorage, and the
    // SDK all authenticate via the user-assigned managed identity. Shared
    // keys are disabled to close the lateral-movement vector and to enforce
    // that any caller (CI, local dev, ops scripts) acquires an Entra token.
    allowSharedKeyAccess: false
    allowBlobPublicAccess: false
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Deployment container used by Flex Consumption when WEBSITE_RUN_FROM_PACKAGE
// flow falls back; harmless to pre-create.
resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {}
}

resource deploymentContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobServices
  name: 'deploymentpackage'
  properties: {
    publicAccess: 'None'
  }
}

// --- Flex Consumption plan ------------------------------------------------

resource plan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: planName
  location: location
  tags: tags
  kind: 'functionapp'
  sku: {
    name: 'FC1'
    tier: 'FlexConsumption'
  }
  properties: {
    reserved: true
  }
}

// --- Function app (Flex Consumption, Python 3.11) -------------------------

resource functionApp 'Microsoft.Web/sites@2023-12-01' = {
  name: functionAppName
  location: location
  // Tag azd-service-name so `azd deploy` knows which Bicep resource hosts the
  // 'function_app' service declared in azure.yaml.
  tags: union(tags, {
    'azd-service-name': 'function_app'
  })
  kind: 'functionapp,linux'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${uamiResourceId}': {}
    }
  }
  properties: {
    serverFarmId: plan.id
    httpsOnly: true
    keyVaultReferenceIdentity: uamiResourceId
    functionAppConfig: {
      deployment: {
        storage: {
          type: 'blobContainer'
          value: '${storage.properties.primaryEndpoints.blob}deploymentpackage'
          authentication: {
            type: 'UserAssignedIdentity'
            userAssignedIdentityResourceId: uamiResourceId
          }
        }
      }
      runtime: {
        name: 'python'
        version: '3.11'
      }
      scaleAndConcurrency: {
        maximumInstanceCount: 100
        instanceMemoryMB: 2048
      }
    }
    siteConfig: {
      appSettings: [
        {
          name: 'AzureWebJobsStorage__accountName'
          value: storage.name
        }
        {
          name: 'AzureWebJobsStorage__credential'
          value: 'managedidentity'
        }
        {
          name: 'AzureWebJobsStorage__clientId'
          value: uamiClientId
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
        {
          name: 'COSMOS_DB__accountEndpoint'
          value: cosmosEndpoint
        }
        {
          name: 'COSMOS_DB__credential'
          value: 'managedidentity'
        }
        {
          name: 'COSMOS_DB__clientId'
          value: uamiClientId
        }
        // --- Plain env vars consumed by function app code ---
        {
          name: 'COSMOS_DB_ENDPOINT'
          value: cosmosEndpoint
        }
        {
          name: 'COSMOS_DB_DATABASE'
          value: cosmosDatabase
        }
        {
          name: 'COSMOS_DB_MEMORIES_CONTAINER'
          value: cosmosContainer
        }
        {
          name: 'COSMOS_DB_TURNS_CONTAINER'
          value: cosmosTurnsContainer
        }
        {
          name: 'COSMOS_DB_SUMMARIES_CONTAINER'
          value: cosmosSummariesContainer
        }
        {
          name: 'COSMOS_DB_LEASE_CONTAINER'
          value: cosmosLeaseContainer
        }
        {
          name: 'COSMOS_DB_COUNTERS_CONTAINER'
          value: cosmosCountersContainer
        }
        {
          name: 'AI_FOUNDRY_ENDPOINT'
          value: aiFoundryEndpoint
        }
        {
          name: 'AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME'
          value: embeddingDeploymentName
        }
        {
          name: 'AI_FOUNDRY_EMBEDDING_DIMENSIONS'
          value: string(embeddingDimensions)
        }
        {
          name: 'AI_FOUNDRY_CHAT_DEPLOYMENT_NAME'
          value: chatDeploymentName
        }
        {
          name: 'AZURE_OPENAI_API_VERSION'
          value: azureOpenAiApiVersion
        }
        {
          name: 'AZURE_CLIENT_ID'
          value: uamiClientId
        }
        {
          name: 'THREAD_SUMMARY_EVERY_N'
          value: string(threadSummaryEveryN)
        }
        {
          name: 'FACT_EXTRACTION_EVERY_N'
          value: string(factExtractionEveryN)
        }
        {
          name: 'USER_SUMMARY_EVERY_N'
          value: string(userSummaryEveryN)
        }
        {
          name: 'DEDUP_EVERY_N'
          value: string(dedupEveryN)
        }
        {
          name: 'MAX_BATCH_SIZE'
          value: string(maxBatchSize)
        }
        {
          name: 'MEMORY_PROCESSOR_OWNER'
          value: memoryProcessorOwner
        }
      ]
    }
  }
  dependsOn: [
    deploymentContainer
  ]
}

// --- Outputs --------------------------------------------------------------

output functionAppName string = functionApp.name
output functionAppUrl string = 'https://${functionApp.properties.defaultHostName}'
output storageAccountName string = storage.name
output appInsightsName string = appInsights.name
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output logAnalyticsName string = logAnalytics.name
