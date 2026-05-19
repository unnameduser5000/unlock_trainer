package com.example.sid_coordinator

data class AdminMutationResult(
    val action: String,
    val success: Boolean,
    val message: String,
    val status: AdminStatusSnapshot
)
