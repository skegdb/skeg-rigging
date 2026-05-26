//! End-to-end check: open tenant, insert, flush, reopen, query.

use skeg_rigging::prelude::*;
use skeg_rigging_skeg::Tenant;

fn unit(at: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[at] = 1.0;
    v
}

#[test]
fn write_then_read_persists_records() {
    let dir = tempfile::tempdir().unwrap();
    let id = TenantId::from_bytes([1; 16]);

    {
        let t = Tenant::open(dir.path(), id, 4).unwrap();
        t.insert(
            RecordId(1),
            unit(0, 4),
            true,
            vec!["a".into()],
            b"payload-1".to_vec(),
        )
        .unwrap();
        t.insert(
            RecordId(2),
            unit(1, 4),
            false,
            vec!["b".into()],
            b"payload-2".to_vec(),
        )
        .unwrap();
        t.flush().unwrap();
    }

    let t2 = Tenant::open_readonly_at(dir.path()).unwrap();
    assert_eq!(<Tenant as IterVectors>::record_count(&t2), 2);
    assert_eq!(<Tenant as IterVectors>::embedding_dim(&t2), 4);

    let hits = t2
        .query_filtered(&unit(0, 4), 10, &|m: &RecordMeta<'_>| m.shareable)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record_id, RecordId(1));
}

#[test]
fn iter_vectors_returns_all() {
    let dir = tempfile::tempdir().unwrap();
    let id = TenantId::from_bytes([2; 16]);
    let t = Tenant::open(dir.path(), id, 3).unwrap();
    t.insert(
        RecordId(10),
        vec![1.0, 0.0, 0.0],
        true,
        vec![],
        b"x".to_vec(),
    )
    .unwrap();
    t.insert(
        RecordId(20),
        vec![0.0, 1.0, 0.0],
        true,
        vec![],
        b"y".to_vec(),
    )
    .unwrap();
    t.insert(
        RecordId(30),
        vec![0.0, 0.0, 1.0],
        true,
        vec![],
        b"z".to_vec(),
    )
    .unwrap();

    let mut ids: Vec<u64> = t.iter_vectors().map(|(rid, _)| rid.0).collect();
    ids.sort();
    assert_eq!(ids, vec![10, 20, 30]);
}

#[test]
fn dim_mismatch_on_reopen_errors() {
    let dir = tempfile::tempdir().unwrap();
    let id = TenantId::from_bytes([3; 16]);
    {
        let t = Tenant::open(dir.path(), id, 4).unwrap();
        t.flush().unwrap();
    }
    match Tenant::open(dir.path(), id, 8) {
        Ok(_) => panic!("expected DimMismatch"),
        Err(e) => assert!(
            matches!(
                e,
                skeg_rigging_skeg::TenantError::DimMismatch {
                    on_disk: 4,
                    requested: 8
                }
            ),
            "got {e:?}"
        ),
    }
}

#[test]
fn read_only_rejects_writes() {
    let dir = tempfile::tempdir().unwrap();
    let id = TenantId::from_bytes([4; 16]);
    {
        let t = Tenant::open(dir.path(), id, 2).unwrap();
        t.insert(RecordId(1), vec![1.0, 0.0], true, vec![], vec![])
            .unwrap();
        t.flush().unwrap();
    }
    let ro = Tenant::open_readonly_at(dir.path()).unwrap();
    let err = ro
        .insert(RecordId(2), vec![0.0, 1.0], true, vec![], vec![])
        .unwrap_err();
    assert!(matches!(err, skeg_rigging_skeg::TenantError::Io(_)));
}
