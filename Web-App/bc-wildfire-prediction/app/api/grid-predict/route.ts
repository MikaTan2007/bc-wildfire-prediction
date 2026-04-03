import { NextResponse } from "next/server";

export async function GET() {
    const baseUrl = process.env.FASTAPI_URL || "http://127.0.0.1:8000";

    try {
        const resp = await fetch(`${baseUrl}/grid-predict`, {
            method: "GET",
            cache: "no-store",
        });

        if (!resp.ok) {
            const text = await resp.text();
            return NextResponse.json(
                { error: `Upstream API Failed: ${resp.status}`, detail: text },
                { status: 502 }
            );
        }

        const data = await resp.json();
        return NextResponse.json(data, { status: 200 });

    } catch (err) {
        return NextResponse.json(
            {
                error: "Failed to reach FastAPI backend",
                detail: err instanceof Error ? err.message : "Unknown error"
            },
            { status: 500 }
        )
    }
}