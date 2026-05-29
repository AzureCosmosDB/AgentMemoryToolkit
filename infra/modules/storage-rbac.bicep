// RBAC for the user-assigned managed identity (and optionally a user
// principalId) on the function app's Storage account.
//
// Storage always lives in the toolkit's own resource group (it's only
// provisioned when deployFunctionApp=true), so this module is always
// scoped locally.
//
// Built-in roles:
//   - b7e6dc6d-f1e8-4753-8033-0f276bb0955b — Storage Blob Data Owner.
//   - 974c5e8b-45b9-4653-ba55-5f855dd0fb88 — Storage Queue Data Contributor.
//   - 0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3 — Storage Table Data Contributor.
//
// Durable Functions (default Azure Storage provider) talks to Storage Queues
// + Tables under the function app's identity. Without those two roles, the
// very first orchestration start returns 403 even though Blob is fine.

@description('Name of the Storage account (must already exist in this module\'s scope).')
param storageAccountName string

@description('Principal ID of the UAMI used by the function app. Empty string skips.')
param functionPrincipalId string = ''

@description('Optional user principal id (the deploying user) for local sample access. Empty string skips.')
param userPrincipalId string = ''

@description('AAD principal type for userPrincipalId. Use ServicePrincipal when deploying from CI under an SP.')
@allowed([
  'User'
  'ServicePrincipal'
  'Group'
])
param userPrincipalType string = 'User'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

var storageBlobDataOwnerRoleId = 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b'
var storageQueueDataContributorRoleId = '974c5e8b-45b9-4653-ba55-5f855dd0fb88'
var storageTableDataContributorRoleId = '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3'

resource storageBlobRoleFunction 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(functionPrincipalId)) {
  scope: storage
  name: guid(storage.id, functionPrincipalId, storageBlobDataOwnerRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataOwnerRoleId)
    principalId: functionPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource storageQueueRoleFunction 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(functionPrincipalId)) {
  scope: storage
  name: guid(storage.id, functionPrincipalId, storageQueueDataContributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageQueueDataContributorRoleId)
    principalId: functionPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource storageTableRoleFunction 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(functionPrincipalId)) {
  scope: storage
  name: guid(storage.id, functionPrincipalId, storageTableDataContributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageTableDataContributorRoleId)
    principalId: functionPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource storageBlobRoleUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(userPrincipalId)) {
  scope: storage
  name: guid(storage.id, userPrincipalId, storageBlobDataOwnerRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataOwnerRoleId)
    principalId: userPrincipalId
    principalType: userPrincipalType
  }
}
