// Agent Memory Toolkit — main entry point.
// Provisions: Cosmos NoSQL serverless + AI Foundry (Cognitive Services
// AIServices, two model deployments) + user-assigned managed identity +
// RBAC + Function app on Flex Consumption (Storage, App Insights, Log
// Analytics, leases + counter containers, storage RBAC).
//
// The Function app is always deployed (Flex Consumption is pay-per-execution
// — idle cost is essentially $0). Customers using the in-process
// MemoryProcessor simply ignore it. An advanced escape hatch exists via the
// `deployFunctionApp` parameter (default `true`); see infra/README.md.

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

@description('Whether to deploy the Function app. Defaults to true. Set false only if you have a strong reason to skip it (Flex Consumption is pay-per-execution — idle cost is ~$0).')
param deployFunctionApp bool = true

@description('Use an existing Cosmos account instead of creating a new one (BYOR).')
param useExistingCosmos bool = false

@description('Existing Cosmos account name (when useExistingCosmos=true).')
param existingCosmosAccountName string = ''

@description('Existing Cosmos account resource group (when useExistingCosmos=true).')
param existingCosmosResourceGroup string = ''

@description('Use an existing AI Foundry / Cognitive Services account instead of creating a new one (BYOR).')
param useExistingAiFoundry bool = false

@description('Existing AI Foundry account name (when useExistingAiFoundry=true).')
param existingAiFoundryName string = ''

@description('Existing AI Foundry account resource group (when useExistingAiFoundry=true).')
param existingAiFoundryResourceGroup string = ''

@description('Cosmos database name.')
param cosmosDatabaseName string = 'ai_memory'

@description('Turns container name.')
param turnsContainerName string = 'memories_turns'

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

// --- Cosmos ---------------------------------------------------------------

module cosmos 'modules/cosmos.bicep' = {
  scope: rg
  name: 'cosmos'
  params: {
    useExisting: useExistingCosmos
    accountName: cosmosAccountName
    existingAccountName: existingCosmosAccountName
    existingResourceGroup: existingCosmosResourceGroup
    location: location
    databaseName: cosmosDatabaseName
    turnsContainerName: turnsContainerName
    memoriesTurnsDefaultTtl: memoriesTurnsDefaultTtl
    deployFunctionContainers: deployFunctionApp
    tags: commonTags
  }
}

// --- AI Foundry (Cognitive Services AIServices) ---------------------------

module aiFoundry 'modules/ai-foundry.bicep' = {
  scope: rg
  name: 'aiFoundry'
  params: {
    useExisting: useExistingAiFoundry
    accountName: aiFoundryAccountName
    existingAccountName: existingAiFoundryName
    existingResourceGroup: existingAiFoundryResourceGroup
    location: location
    chatModelName: chatModelName
    chatDeploymentName: chatDeploymentName
    embeddingModelName: embeddingModelName
    embeddingDeploymentName: embeddingDeploymentName
    tags: commonTags
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
    cosmosLeaseContainer: cosmos.outputs.leasesContainerName
    cosmosCountersContainer: cosmos.outputs.counterContainerName
    aiFoundryEndpoint: aiFoundry.outputs.endpoint
    embeddingDeploymentName: aiFoundry.outputs.embeddingDeploymentName
    chatDeploymentName: aiFoundry.outputs.chatDeploymentName
    tags: commonTags
  }
}

// --- RBAC -----------------------------------------------------------------
// Granted to:
//   - The function app's user-assigned managed identity (always; harmless when
//     the function app isn't deployed yet because nothing is using it).
//   - The deploying user (when principalId is supplied) so they can hit the
//     Cosmos data plane and AI Foundry from local samples.

module rbac 'modules/rbac.bicep' = {
  scope: rg
  name: 'rbac'
  params: {
    cosmosAccountName: cosmos.outputs.accountName
    aiFoundryAccountName: aiFoundry.outputs.accountName
    storageAccountName: deployFunctionApp ? functions!.outputs.storageAccountName : ''
    functionPrincipalId: identity.outputs.principalId
    userPrincipalId: principalId
  }
}

// --- Outputs (consumed by azd → .azure/<env>/.env) ------------------------

output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = subscription().tenantId
output RESOURCE_GROUP_NAME string = rg.name

output COSMOS_DB_ENDPOINT string = cosmos.outputs.endpoint
output COSMOS_DB_DATABASE string = cosmos.outputs.databaseName
output COSMOS_DB_CONTAINER string = cosmos.outputs.memoriesContainerName
output COSMOS_TURNS_CONTAINER string = cosmos.outputs.turnsContainerName
output COSMOS_DB_ACCOUNT_NAME string = cosmos.outputs.accountName

output AI_FOUNDRY_ENDPOINT string = aiFoundry.outputs.endpoint
output AI_FOUNDRY_ACCOUNT_NAME string = aiFoundry.outputs.accountName
output AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME string = aiFoundry.outputs.embeddingDeploymentName
output AI_FOUNDRY_CHAT_DEPLOYMENT_NAME string = aiFoundry.outputs.chatDeploymentName

output AZURE_CLIENT_ID string = identity.outputs.clientId
output AZURE_USER_ASSIGNED_IDENTITY_ID string = identity.outputs.id

output FUNCTION_APP_NAME string = deployFunctionApp ? functions!.outputs.functionAppName : ''
output FUNCTION_APP_URL string = deployFunctionApp ? functions!.outputs.functionAppUrl : ''
