// RBAC for the user-assigned managed identity (and optionally a user
// principalId) on a single AI Foundry / Cognitive Services account.
//
// This module must be scoped to the resource group that contains the AI
// Foundry account — even when that's not the toolkit's own RG.
//
// Built-in role:
//   - 5e0bd9bd-7b93-4f28-af87-19fc36ad61bd — Cognitive Services OpenAI User.

@description('Name of the AI Foundry account (must already exist in this module\'s scope).')
param aiFoundryAccountName string

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

resource aiFoundry 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: aiFoundryAccountName
}

var cognitiveServicesOpenAIUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

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
    principalType: userPrincipalType
  }
}
