// AI Foundry account (Cognitive Services kind=AIServices) + 2 model deployments.
//
// Implementation choice:
//   We use a single Microsoft.CognitiveServices/accounts resource with kind=AIServices
//   instead of the full AI Foundry hub+project (which would also require a
//   Storage account, Key Vault, Application Insights, and an ML workspace).
//   The AIServices account exposes the same Azure OpenAI–compatible endpoint
//   (https://<name>.cognitiveservices.azure.com/) and supports Entra-based
//   "Cognitive Services OpenAI User" RBAC, which is exactly what the toolkit
//   needs for embedding and chat completion calls. This is the pattern used by
//   most current Azure-Samples azd templates (e.g. azure-search-openai-demo,
//   openai-chat-app-quickstart).

@description('Whether to use an existing AI Foundry / Cognitive Services account.')
param useExisting bool = false

@description('Name of the new AI Foundry account (when useExisting=false).')
param accountName string

@description('Name of an existing account (when useExisting=true).')
param existingAccountName string = ''

@description('Resource group of the existing account (when useExisting=true). If empty, current RG is used.')
param existingResourceGroup string = ''

@description('Azure region. Pin to one with required model availability (eastus2, swedencentral, westus3).')
@allowed([
  'eastus2'
  'swedencentral'
  'westus3'
  'eastus'
])
param location string = 'eastus2'

@description('Catalog name of the chat completion model (e.g. gpt-4o-mini).')
param chatModelName string = 'gpt-4o-mini'

@description('Chat model version.')
param chatModelVersion string = '2024-07-18'

@description('Deployment name to expose the chat model under. Defaults to the model name when empty.')
param chatDeploymentName string = ''

@description('Chat model SKU capacity (TPM, in thousands).')
param llmCapacity int = 30

@description('Catalog name of the embedding model (e.g. text-embedding-3-large).')
param embeddingModelName string = 'text-embedding-3-large'

@description('Embedding model version.')
param embeddingModelVersion string = '1'

@description('Deployment name to expose the embedding model under. Defaults to the model name when empty.')
param embeddingDeploymentName string = ''

@description('Embedding model SKU capacity (TPM, in thousands).')
param embeddingCapacity int = 30

@description('Tags to apply to created resources.')
param tags object = {}

// --- Account ---------------------------------------------------------------

resource newAccount 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (!useExisting) {
  name: accountName
  location: location
  tags: tags
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: accountName
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
  }
}

resource existingAccount 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = if (useExisting) {
  name: existingAccountName
  scope: resourceGroup(empty(existingResourceGroup) ? resourceGroup().name : existingResourceGroup)
}

var effectiveAccountName = useExisting ? existingAccountName : accountName
var manageDeployments = !useExisting || empty(existingResourceGroup)
var effectiveChatDeploymentName = empty(chatDeploymentName) ? chatModelName : chatDeploymentName
var effectiveEmbeddingDeploymentName = empty(embeddingDeploymentName) ? embeddingModelName : embeddingDeploymentName

resource accountRef 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = if (manageDeployments) {
  name: effectiveAccountName
  dependsOn: [
    newAccount
  ]
}

// --- Model deployments -----------------------------------------------------
// Note: deployments are serialized via dependsOn — Cognitive Services rejects
// concurrent deployment writes on the same account.

resource llmDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (manageDeployments) {
  parent: accountRef
  name: effectiveChatDeploymentName
  sku: {
    name: 'GlobalStandard'
    capacity: llmCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: chatModelName
      version: chatModelVersion
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    raiPolicyName: 'Microsoft.DefaultV2'
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (manageDeployments) {
  parent: accountRef
  name: effectiveEmbeddingDeploymentName
  sku: {
    name: 'Standard'
    capacity: embeddingCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: embeddingModelName
      version: embeddingModelVersion
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    raiPolicyName: 'Microsoft.DefaultV2'
  }
  dependsOn: [
    llmDeployment
  ]
}

// --- Outputs ---------------------------------------------------------------

output accountName string = effectiveAccountName
output endpoint string = useExisting ? existingAccount!.properties.endpoint : newAccount!.properties.endpoint
output accountResourceId string = useExisting ? existingAccount!.id : newAccount!.id
output chatDeploymentName string = effectiveChatDeploymentName
output embeddingDeploymentName string = effectiveEmbeddingDeploymentName
