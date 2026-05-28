// AI Foundry (Cognitive Services kind=AIServices) account + chat + embedding
// model deployments. Single-file module — no `existing` keyword tricks because
// the account is always created fresh by this template.
//
// We use a single Microsoft.CognitiveServices/accounts resource with
// kind=AIServices instead of the full AI Foundry hub+project (which would
// also require Storage, Key Vault, App Insights, and an ML workspace). The
// AIServices account exposes the Azure OpenAI–compatible endpoint
// (https://<name>.cognitiveservices.azure.com/) and supports Entra-based
// "Cognitive Services OpenAI User" RBAC, which is what the toolkit needs.

@description('Name of the AI Foundry account to create.')
param accountName string

@description('Azure region. Pin to one with required model availability.')
@allowed([
  'eastus2'
  'swedencentral'
  'westus3'
  'eastus'
])
param location string = 'eastus2'

@description('Tags to apply to the account.')
param tags object = {}

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

var effectiveChatDeploymentName = empty(chatDeploymentName) ? chatModelName : chatDeploymentName
var effectiveEmbeddingDeploymentName = empty(embeddingDeploymentName) ? embeddingModelName : embeddingDeploymentName

// --- Account --------------------------------------------------------------

resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
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

// --- Model deployments ----------------------------------------------------
//
// Deployments are serialized via dependsOn — Cognitive Services rejects
// concurrent deployment writes on the same account.

resource llmDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: account
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

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: account
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

// --- Outputs --------------------------------------------------------------

output accountName string = account.name
output accountResourceId string = account.id
output endpoint string = account.properties.endpoint
output chatDeploymentName string = effectiveChatDeploymentName
output embeddingDeploymentName string = effectiveEmbeddingDeploymentName
