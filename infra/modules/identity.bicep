// User-assigned managed identity used by the Function app and granted RBAC
// to Cosmos, AI Foundry, and Storage by rbac.bicep.

@description('Name of the UAMI to create.')
param name string

@description('Azure region.')
param location string

@description('Tags to apply.')
param tags object = {}

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: name
  location: location
  tags: tags
}

output id string = uami.id
output name string = uami.name
output principalId string = uami.properties.principalId
output clientId string = uami.properties.clientId
