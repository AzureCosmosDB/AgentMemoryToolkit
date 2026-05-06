// RBAC for the user-assigned managed identity (and optionally a user
// principalId) on Cosmos DB, AI Foundry / Cognitive Services, and Storage.
//
// - Cosmos DB Built-in Data Contributor (data-plane SQL role) on the Cosmos
//   account scope, assigned via sqlRoleAssignments.
// - Cognitive Services OpenAI User on the AI Foundry account.
// - Storage Blob Data Owner on the Storage account (only when a storage account
//   id is supplied — i.e. when the function app is deployed).

@description('Cosmos account name (used for nested role assignments).')
param cosmosAccountName string

@description('AI Foundry / Cognitive Services account name.')
param aiFoundryAccountName string

@description('Storage account name. Empty string disables storage RBAC (sdk-only profile).')
param storageAccountName string = ''

@description('Principal ID of the UAMI used by the function app.')
param functionPrincipalId string = ''

@description('Optional user principal id (the deploying user) for data-plane access during testing. Empty string skips.')
param userPrincipalId string = ''

// --- existing resources ----------------------------------------------------

resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' existing = {
  name: cosmosAccountName
}

resource aiFoundry 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: aiFoundryAccountName
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = if (!empty(storageAccountName)) {
  name: storageAccountName
}

// --- Built-in role IDs (subscription scope) --------------------------------
// https://learn.microsoft.com/azure/role-based-access-control/built-in-roles

var cognitiveServicesOpenAIUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
var storageBlobDataOwnerRoleId = 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b'
// Durable Functions task hub (default Azure Storage provider) talks to
// Storage Queues + Tables under the function app's identity. Without these,
// the very first orchestration start returns 403 even though Blob is fine.
var storageQueueDataContributorRoleId = '974c5e8b-45b9-4653-ba55-5f855dd0fb88'
var storageTableDataContributorRoleId = '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3'

// --- Cosmos data-plane role assignment -------------------------------------
// Cosmos DB Built-in Data Contributor (00000000-0000-0000-0000-000000000002).

var cosmosDataContributorRoleId = '00000000-0000-0000-0000-000000000002'
var cosmosScope = cosmos.id

resource cosmosRoleFunction 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = if (!empty(functionPrincipalId)) {
  parent: cosmos
  name: guid(cosmos.id, functionPrincipalId, cosmosDataContributorRoleId)
  properties: {
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/${cosmosDataContributorRoleId}'
    principalId: functionPrincipalId
    scope: cosmosScope
  }
}

resource cosmosRoleUser 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = if (!empty(userPrincipalId)) {
  parent: cosmos
  name: guid(cosmos.id, userPrincipalId, cosmosDataContributorRoleId)
  properties: {
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/${cosmosDataContributorRoleId}'
    principalId: userPrincipalId
    scope: cosmosScope
  }
}

// --- AI Foundry role assignments -------------------------------------------

resource aiFoundryRoleFunction 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(functionPrincipalId)) {
  scope: aiFoundry
  name: guid(aiFoundry.id, functionPrincipalId, cognitiveServicesOpenAIUserRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRoleId)
    principalId: functionPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource aiFoundryRoleUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(userPrincipalId)) {
  scope: aiFoundry
  name: guid(aiFoundry.id, userPrincipalId, cognitiveServicesOpenAIUserRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRoleId)
    principalId: userPrincipalId
    principalType: 'User'
  }
}

// --- Storage role assignments (function app only) --------------------------

resource storageRoleFunction 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(storageAccountName) && !empty(functionPrincipalId)) {
  scope: storage
  name: guid(storage.id, functionPrincipalId, storageBlobDataOwnerRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataOwnerRoleId)
    principalId: functionPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource storageQueueRoleFunction 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(storageAccountName) && !empty(functionPrincipalId)) {
  scope: storage
  name: guid(storage.id, functionPrincipalId, storageQueueDataContributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageQueueDataContributorRoleId)
    principalId: functionPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource storageTableRoleFunction 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(storageAccountName) && !empty(functionPrincipalId)) {
  scope: storage
  name: guid(storage.id, functionPrincipalId, storageTableDataContributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageTableDataContributorRoleId)
    principalId: functionPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource storageRoleUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(storageAccountName) && !empty(userPrincipalId)) {
  scope: storage
  name: guid(storage.id, userPrincipalId, storageBlobDataOwnerRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataOwnerRoleId)
    principalId: userPrincipalId
    principalType: 'User'
  }
}
