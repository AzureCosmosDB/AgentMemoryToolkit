// RBAC for the user-assigned managed identity (and optionally a user
// principalId) on a single Cosmos DB account.
//
// Cosmos data-plane access is granted via sqlRoleAssignments (children of
// the account), so this module must be scoped to the resource group that
// contains the Cosmos account — even when that's not the toolkit's own RG.
//
// Built-in roles:
//   - 00000000-0000-0000-0000-000000000001 — Cosmos DB Built-in Data Reader.
//   - 00000000-0000-0000-0000-000000000002 — Cosmos DB Built-in Data Contributor.
//
// Both roles are granted so the principal has explicit read-only access in
// addition to read/write. Useful for downstream consumers (audit dashboards,
// analytics jobs) that should run as the same identity but are validated by
// security review as needing only the Reader scope.

@description('Name of the Cosmos account (must already exist in this module\'s scope).')
param cosmosAccountName string

@description('Principal ID of the UAMI used by the function app. Empty string skips.')
param functionPrincipalId string = ''

@description('Optional user principal id (the deploying user) for local sample access. Empty string skips.')
param userPrincipalId string = ''

resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' existing = {
  name: cosmosAccountName
}

var cosmosDataReaderRoleId = '00000000-0000-0000-0000-000000000001'
var cosmosDataContributorRoleId = '00000000-0000-0000-0000-000000000002'

resource cosmosReaderRoleFunction 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = if (!empty(functionPrincipalId)) {
  parent: cosmos
  name: guid(cosmos.id, functionPrincipalId, cosmosDataReaderRoleId)
  properties: {
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/${cosmosDataReaderRoleId}'
    principalId: functionPrincipalId
    scope: cosmos.id
  }
}

resource cosmosContributorRoleFunction 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = if (!empty(functionPrincipalId)) {
  parent: cosmos
  name: guid(cosmos.id, functionPrincipalId, cosmosDataContributorRoleId)
  properties: {
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/${cosmosDataContributorRoleId}'
    principalId: functionPrincipalId
    scope: cosmos.id
  }
}

resource cosmosReaderRoleUser 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = if (!empty(userPrincipalId)) {
  parent: cosmos
  name: guid(cosmos.id, userPrincipalId, cosmosDataReaderRoleId)
  properties: {
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/${cosmosDataReaderRoleId}'
    principalId: userPrincipalId
    scope: cosmos.id
  }
}

resource cosmosContributorRoleUser 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = if (!empty(userPrincipalId)) {
  parent: cosmos
  name: guid(cosmos.id, userPrincipalId, cosmosDataContributorRoleId)
  properties: {
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/${cosmosDataContributorRoleId}'
    principalId: userPrincipalId
    scope: cosmos.id
  }
}
