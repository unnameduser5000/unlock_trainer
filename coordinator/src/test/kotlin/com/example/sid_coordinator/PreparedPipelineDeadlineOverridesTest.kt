package com.example.sid_coordinator

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PreparedPipelineDeadlineOverridesTest {
    @Test
    fun parsesPerRecordDeadlines() {
        assertEquals(
            mapOf(15 to 1_000L, 31 to 2_500L),
            parseSubmitRpcDeadlineOverrides("15:1000,31:2500")
        )
    }

    @Test
    fun acceptsMissingOrNoneAsNoOverrides() {
        assertEquals(emptyMap(), parseSubmitRpcDeadlineOverrides(null))
        assertEquals(emptyMap(), parseSubmitRpcDeadlineOverrides("none"))
    }

    @Test
    fun rejectsMalformedOrDuplicateOverrides() {
        assertFailsWith<IllegalArgumentException> { parseSubmitRpcDeadlineOverrides("15") }
        assertFailsWith<IllegalArgumentException> { parseSubmitRpcDeadlineOverrides("-1:1000") }
        assertFailsWith<IllegalArgumentException> { parseSubmitRpcDeadlineOverrides("15:0") }
        assertFailsWith<IllegalArgumentException> { parseSubmitRpcDeadlineOverrides("15:1000,15:2000") }
    }
}
