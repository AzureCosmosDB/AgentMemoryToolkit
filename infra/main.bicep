// Agent Memory Toolkit — main entry point.
// See infra/README.md for architecture and operational knobs.

targetScope = 'subscription'

// --- Parameters -----------------------------------------------------------

@minLength(1)
@maxLength(64)
@description('Name of the azd environment. Used as the resource-group suffix and as a unique-name token seed.')
param environmentName string

@description('Default region. Pinned to one with Cosmos serverless + AI Foundry + model availability.')
@allowed([
  'eastus2'
  'swedencentral'
  'westus3'
  'eastus'
])
param location string = 'eastus2'

@description('Object id of the user running azd. Used to grant data-plane RBAC (Cosmos, AI Foundry, Storage) so the deployer can run samples locally.')
param principalId string = ''

@description('AAD principal type for principalId. Defaults to User; set to ServicePrincipal when deploying from CI under an SP to avoid PrincipalTypeMismatch on AI Foundry / Storage RBAC.')
@allowed([
  'User'
  'ServicePrincipal'
  'Group'
])
param principalType string = 'User'

@description('Whether to deploy the Function app. Defaults to true. Set false only if you have a strong reason to skip it (Flex Consumption is pay-per-execution — idle cost is ~$0).')
param deployFunctionApp bool = true

@description('Cosmos database name.')
param cosmosDatabaseName string = 'ai_memory'

@description('Memories container name.')
param memoriesContainerName string = 'memories'

@description('Turns container name.')
param turnsContainerName string = 'memories_turns'

@description('Summaries container name.')
param summariesContainerName string = 'memories_summaries'

@description('Default TTL for turn documents, in seconds. Use -1 to disable expiry.')
param memoriesTurnsDefaultTtl int = 2592000

@description('Catalog name of the embedding model (e.g. text-embedding-3-large).')
param embeddingModelName string = 'text-embedding-3-large'

@description('Deployment name to expose the embedding model under. Defaults to the model name when empty.')
param embeddingDeploymentName string = ''

@description('Catalog name of the chat completion model (e.g. gpt-4o-mini).')
param chatModelName string = 'gpt-4o-mini'

@description('Deployment name to expose the chat model under. Defaults to the model name when empty.')
param chatDeploymentName string = ''

@description('Azure OpenAI REST API version pinned for both chat and embedding clients (SDK + function-app). Newer preview versions are required for strict JSON-schema response_format on gpt-5.x models.')
param azureOpenAiApiVersion string = '2024-12-01-preview'

@description('Embedding output dimensions. MUST equal the dimensions configured on the Cosmos memories container vectorEmbeddingPolicy. text-embedding-3-large natively returns 3072; we set 1536 here so DiskANN (also 1536 in cosmos.bicep) can match. Change this only if you also change cosmos.bicep.')
param embeddingDimensions int = 1536

@description('Run thread-summary orchestration every N turns within a (user_id, thread_id). 0 = disabled.')
param threadSummaryEveryN int = 10

@description('Run extract-memories every N change-feed batches. Default 1 = run on every batch (matches SDK + local template). Bump for cost-sensitive production deployments.')
param factExtractionEveryN int = 1

@description('Run dedup once per N fact-extraction batches. Effective cadence = factExtractionEveryN * dedupEveryN turns.')
param dedupEveryN int = 5

@description('Run user-summary orchestration every N turns from a given user_id across all threads. 0 = disabled.')
param userSummaryEveryN int = 20

@description('Maximum number of change-feed items processed per orchestration batch.')
param maxBatchSize int = 20

@description('Backend that owns processing. `durable` (default) = the function-app fleet owns processing; SDK clients pointed at the same container will skip auto-triggering. `inprocess` = SDK owns processing, function-app skips.')
@allowed([
  'durable'
  'inprocess'
])
param memoryProcessorOwner string = 'durable'

// --- Naming ---------------------------------------------------------------

var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = take(uniqueString(subscription().id, environmentName), 13)

var resourceGroupName = '${abbrs.resourceGroup}${environmentName}'
var cosmosAccountName = '${abbrs.cosmosAccount}${resourceToken}'
var aiFoundryAccountName = '${abbrs.aiFoundryAccount}${resourceToken}'
var uamiName = '${abbrs.userAssignedIdentity}${resourceToken}'
var functionAppName = '${abbrs.functionApp}${resourceToken}'
var storageAccountName = take(toLower('${abbrs.storageAccount}${resourceToken}'), 24)
var appInsightsName = '${abbrs.appInsights}${resourceToken}'
var logAnalyticsName = '${abbrs.logAnalytics}${resourceToken}'
var planName = '${abbrs.appServicePlan}${resourceToken}'

var commonTags = {
  'azd-env-name': environmentName
  workload: 'agent-memory-toolkit'
}

// --- Resource group -------------------------------------------------------

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: commonTags
}

// --- Identity -------------------------------------------------------------

module identity 'modules/identity.bicep' = {
  scope: rg
  name: 'identity'
  params: {
    name: uamiName
    location: location
    tags: commonTags
  }
}

// --- Cosmos (account + database + containers) -----------------------------

module cosmos 'modules/cosmos.bicep' = {
  scope: rg
  name: 'cosmos'
  params: {
    accountName: cosmosAccountName
    location: location
    tags: commonTags
    databaseName: cosmosDatabaseName
    memoriesContainerName: memoriesContainerName
    turnsContainerName: turnsContainerName
    summariesContainerName: summariesContainerName
    memoriesTurnsDefaultTtl: memoriesTurnsDefaultTtl
    deployFunctionContainers: deployFunctionApp
    embeddingDimensions: embeddingDimensions
  }
}

// --- AI Foundry (account + model deployments) -----------------------------

module aiFoundry 'modules/ai-foundry.bicep' = {
  scope: rg
  name: 'ai-foundry'
  params: {
    accountName: aiFoundryAccountName
    location: location
    tags: commonTags
    chatModelName: chatModelName
    chatDeploymentName: chatDeploymentName
    embeddingModelName: embeddingModelName
    embeddingDeploymentName: embeddingDeploymentName
  }
}

// --- Function app (full profile only) -------------------------------------

module functions 'modules/functions.bicep' = if (deployFunctionApp) {
  scope: rg
  name: 'functions'
  params: {
    functionAppName: functionAppName
    storageAccountName: storageAccountName
    appInsightsName: appInsightsName
    logAnalyticsName: logAnalyticsName
    planName: planName
    location: location
    uamiResourceId: identity.outputs.id
    uamiClientId: identity.outputs.clientId
    cosmosEndpoint: cosmos.outputs.endpoint
    cosmosDatabase: cosmos.outputs.databaseName
    cosmosContainer: cosmos.outputs.memoriesContainerName
    cosmosTurnsContainer: cosmos.outputs.turnsContainerName
    cosmosSummariesContainer: cosmos.outputs.summariesContainerName
    cosmosLeaseContainer: cosmos.outputs.leasesContainerName
    cosmosCountersContainer: cosmos.outputs.counterContainerName
    aiFoundryEndpoint: aiFoundry.outputs.endpoint
    embeddingDeploymentName: aiFoundry.outputs.embeddingDeploymentName
    chatDeploymentName: aiFoundry.outputs.chatDeploymentName
    azureOpenAiApiVersion: azureOpenAiApiVersion
    embeddingDimensions: embeddingDimensions
    threadSummaryEveryN: threadSummaryEveryN
    factExtractionEveryN: factExtractionEveryN
    dedupEveryN: dedupEveryN
    userSummaryEveryN: userSummaryEveryN
    maxBatchSize: maxBatchSize
    memoryProcessorOwner: memoryProcessorOwner
    tags: commonTags
  }
}

// --- RBAC -----------------------------------------------------------------
//
// Granted to:
//   - The function app's user-assigned managed identity (always; harmless
//     when the function app isn't deployed because nothing is using it).
//   - The deploying user (when principalId is supplied) so they can hit the
//     Cosmos data plane and AI Foundry from local samples.

module cosmosRbac 'modules/cosmos-rbac.bicep' = {
  scope: rg
  name: 'cosmos-rbac'
  params: {
    cosmosAccountName: cosmos.outputs.accountName
    functionPrincipalId: identity.outputs.principalId
    userPrincipalId: principalId
  }
}

module aiFoundryRbac 'modules/ai-foundry-rbac.bicep' = {
  scope: rg
  name: 'ai-foundry-rbac'
  params: {
    aiFoundryAccountName: aiFoundry.outputs.accountName
    functionPrincipalId: identity.outputs.principalId
    userPrincipalId: principalId
    userPrincipalType: principalType
  }
}

module storageRbac 'modules/storage-rbac.bicep' = if (deployFunctionApp) {
  scope: rg
  name: 'storage-rbac'
  params: {
    storageAccountName: storageAccountName
    functionPrincipalId: identity.outputs.principalId
    userPrincipalId: principalId
    userPrincipalType: principalType
  }
  dependsOn: [
    functions
  ]
}

// --- Outputs (consumed by azd → .azure/<env>/.env) ------------------------

output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = subscription().tenantId
output RESOURCE_GROUP_NAME string = rg.name

output COSMOS_DB_ENDPOINT string = cosmos.outputs.endpoint
output COSMOS_DB_DATABASE string = cosmos.outputs.databaseName
output COSMOS_DB_MEMORIES_CONTAINER string = cosmos.outputs.memoriesContainerName
output COSMOS_DB_TURNS_CONTAINER string = cosmos.outputs.turnsContainerName
output COSMOS_DB_SUMMARIES_CONTAINER string = cosmos.outputs.summariesContainerName
output COSMOS_DB_ACCOUNT_NAME string = cosmos.outputs.accountName

output AI_FOUNDRY_ENDPOINT string = aiFoundry.outputs.endpoint
output AI_FOUNDRY_ACCOUNT_NAME string = aiFoundry.outputs.accountName
output AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME string = aiFoundry.outputs.embeddingDeploymentName
output AI_FOUNDRY_CHAT_DEPLOYMENT_NAME string = aiFoundry.outputs.chatDeploymentName

output AZURE_CLIENT_ID string = identity.outputs.clientId
output AZURE_USER_ASSIGNED_IDENTITY_ID string = identity.outputs.id

output FUNCTION_APP_NAME string = deployFunctionApp ? functions!.outputs.functionAppName : ''
output FUNCTION_APP_URL string = deployFunctionApp ? functions!.outputs.functionAppUrl : ''

output MEMORY_PROCESSOR_OWNER string = deployFunctionApp ? memoryProcessorOwner : 'inprocess'
